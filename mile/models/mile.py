import torch
import torch.nn as nn
import timm

from mile.constants import CARLA_FPS, DISPLAY_SEGMENTATION
from mile.utils.network_utils import pack_sequence_dim, unpack_sequence_dim, remove_past
from mile.models.common import BevDecoder, Decoder, RouteEncode, Policy
from mile.models.frustum_pooling import FrustumPooling
from mile.layers.layers import BasicBlock
from mile.models.transition import RSSM


class Mile(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.receptive_field = cfg.RECEPTIVE_FIELD

        # Image feature encoder
        if self.cfg.MODEL.ENCODER.NAME == 'resnet18':
            self.encoder = timm.create_model(
                cfg.MODEL.ENCODER.NAME, pretrained=True, features_only=True, out_indices=[2, 3, 4],
            )
            feature_info = self.encoder.feature_info.get_dicts(keys=['num_chs', 'reduction'])

        self.feat_decoder = Decoder(feature_info, self.cfg.MODEL.ENCODER.OUT_CHANNELS)

        if not self.cfg.EVAL.NO_LIFTING:
            # Frustum pooling
            bev_downsample = cfg.BEV.FEATURE_DOWNSAMPLE
            self.frustum_pooling = FrustumPooling(
                size=(cfg.BEV.SIZE[0] // bev_downsample, cfg.BEV.SIZE[1] // bev_downsample),
                scale=cfg.BEV.RESOLUTION * bev_downsample,
                offsetx=cfg.BEV.OFFSET_FORWARD / bev_downsample,
                dbound=cfg.BEV.FRUSTUM_POOL.D_BOUND,
                downsample=8,
            )

            # mono depth head
            self.depth_decoder = Decoder(feature_info, self.cfg.MODEL.ENCODER.OUT_CHANNELS)
            self.depth = nn.Conv2d(self.depth_decoder.out_channels, self.frustum_pooling.D, kernel_size=1)
            # only lift argmax of depth distribution for speed
            self.sparse_depth = cfg.BEV.FRUSTUM_POOL.SPARSE
            self.sparse_depth_count = cfg.BEV.FRUSTUM_POOL.SPARSE_COUNT

        backbone_bev_in_channels = self.cfg.MODEL.ENCODER.OUT_CHANNELS

        # Route map
        if self.cfg.MODEL.ROUTE.ENABLED:
            self.backbone_route = RouteEncode(cfg.MODEL.ROUTE.CHANNELS, cfg.MODEL.ROUTE.BACKBONE)
            backbone_bev_in_channels += self.backbone_route.out_channels

        # Measurements
        if self.cfg.MODEL.MEASUREMENTS.ENABLED:
            self.command_encoder = nn.Sequential(
                nn.Embedding(6, self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS),
                nn.Linear(self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS, self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS),
                nn.ReLU(True),
                nn.Linear(self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS, self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS),
                nn.ReLU(True),
            )

            self.command_next_encoder = nn.Sequential(
                nn.Embedding(6, self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS),
                nn.Linear(self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS, self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS),
                nn.ReLU(True),
                nn.Linear(self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS, self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS),
                nn.ReLU(True),
            )

            self.gps_encoder = nn.Sequential(
                nn.Linear(2*2, self.cfg.MODEL.MEASUREMENTS.GPS_CHANNELS),
                nn.ReLU(True),
                nn.Linear(self.cfg.MODEL.MEASUREMENTS.GPS_CHANNELS, self.cfg.MODEL.MEASUREMENTS.GPS_CHANNELS),
                nn.ReLU(True),
            )

            backbone_bev_in_channels += 2*self.cfg.MODEL.MEASUREMENTS.COMMAND_CHANNELS
            backbone_bev_in_channels += self.cfg.MODEL.MEASUREMENTS.GPS_CHANNELS

        # Speed as input
        self.speed_enc = nn.Sequential(
            nn.Linear(1, cfg.MODEL.SPEED.CHANNELS),
            nn.ReLU(True),
            nn.Linear(cfg.MODEL.SPEED.CHANNELS, cfg.MODEL.SPEED.CHANNELS),
            nn.ReLU(True),
        )
        backbone_bev_in_channels += cfg.MODEL.SPEED.CHANNELS
        self.speed_normalisation = cfg.SPEED.NORMALISATION

        # Bev network
        self.backbone_bev = timm.create_model(
            cfg.MODEL.BEV.BACKBONE,
            in_chans=backbone_bev_in_channels,
            pretrained=True,
            features_only=True,
            out_indices=[3],
        )
        feature_info_bev = self.backbone_bev.feature_info.get_dicts(keys=['num_chs', 'reduction'])
        embedding_n_channels = self.cfg.MODEL.EMBEDDING_DIM
        self.final_state_conv = nn.Sequential(
            BasicBlock(feature_info_bev[-1]['num_chs'], embedding_n_channels, stride=2, downsample=True),
            BasicBlock(embedding_n_channels, embedding_n_channels),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
        )

        # Recurrent model
        self.receptive_field = self.cfg.RECEPTIVE_FIELD
        if self.cfg.MODEL.TRANSITION.ENABLED:
            # Recurrent state sequence module
            self.rssm = RSSM(
                embedding_dim=embedding_n_channels,
                action_dim=self.cfg.MODEL.ACTION_DIM,
                hidden_state_dim=self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM,
                state_dim=self.cfg.MODEL.TRANSITION.STATE_DIM,
                action_latent_dim=self.cfg.MODEL.TRANSITION.ACTION_LATENT_DIM,
                receptive_field=self.receptive_field,
                use_dropout=self.cfg.MODEL.TRANSITION.USE_DROPOUT,
                dropout_probability=self.cfg.MODEL.TRANSITION.DROPOUT_PROBABILITY,
            )

        # Policy
        if self.cfg.MODEL.TRANSITION.ENABLED:
            state_dim = self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM + self.cfg.MODEL.TRANSITION.STATE_DIM
        else:
            state_dim = embedding_n_channels
        self.policy = Policy(in_channels=state_dim)

        # Bird's-eye view semantic segmentation
        if self.cfg.SEMANTIC_SEG.ENABLED:
            self.bev_decoder = BevDecoder(
                latent_n_channels=state_dim,
                semantic_n_channels=self.cfg.SEMANTIC_SEG.N_CHANNELS,
            )

        # RGB reconstruction
        if self.cfg.EVAL.RGB_SUPERVISION:
            self.rgb_decoder = BevDecoder(
                latent_n_channels=state_dim,
                semantic_n_channels=3,
                constant_size=(5, 13),
                is_segmentation=False,
            )

        # Used during deployment to save last state
        self.last_h = None
        self.last_sample = None
        self.last_action = None
        self.count = 0

    def forward(self, batch, deployment=False):
        """
        Parameters
        ----------
            batch: dict of torch.Tensor
                keys:
                    image: (b, s, 3, h, w)
                    route_map: (b, s, 3, h_r, w_r)
                    speed: (b, s, 1)
                    intrinsics: (b, s, 3, 3)
                    extrinsics: (b, s, 4, 4)
                    throttle_brake: (b, s, 1)
                    steering: (b, s, 1)
        """
        # Encode RGB images, route_map, speed using intrinsics and extrinsics
        # to a 512 dimensional vector
        b, s = batch['image'].shape[:2]
        embedding = self.encode(batch)  # dim (b, s, 512)

        output = dict()
        if self.cfg.MODEL.TRANSITION.ENABLED:
            # Recurrent state sequence module
            if deployment:
                action = batch['action']
            else:
                action = torch.cat([batch['throttle_brake'], batch['steering']], dim=-1)
            state_dict = self.rssm(embedding, action, use_sample=not deployment, policy=self.policy)

            if deployment:
                state_dict = remove_past(state_dict, s)
                s = 1

            output = {**output, **state_dict}
            state = torch.cat([state_dict['posterior']['hidden_state'], state_dict['posterior']['sample']], dim=-1)
        else:
            state = embedding

        state = pack_sequence_dim(state)
        output_policy = self.policy(state)
        throttle_brake, steering = torch.split(output_policy, 1, dim=-1)
        output['throttle_brake'] = unpack_sequence_dim(throttle_brake, b, s)
        output['steering'] = unpack_sequence_dim(steering, b, s)

        if self.cfg.SEMANTIC_SEG.ENABLED:
            if (not deployment) or (deployment and DISPLAY_SEGMENTATION):
                bev_decoder_output = self.bev_decoder(state)
                bev_decoder_output = unpack_sequence_dim(bev_decoder_output, b, s)
                output = {**output, **bev_decoder_output}

        if self.cfg.EVAL.RGB_SUPERVISION:
            rgb_decoder_output = self.rgb_decoder(state)
            rgb_decoder_output = unpack_sequence_dim(rgb_decoder_output, b, s)
            output = {**output, **rgb_decoder_output}

        return output

    def encode(self, batch):
        b, s = batch['image'].shape[:2]
        image = pack_sequence_dim(batch['image'])
        speed = pack_sequence_dim(batch['speed'])
        intrinsics = pack_sequence_dim(batch['intrinsics'])
        extrinsics = pack_sequence_dim(batch['extrinsics'])

        # Image encoder, multiscale
        xs = self.encoder(image)

        # Lift features to bird's-eye view.
        # Aggregate features to output resolution (H/8, W/8)
        x = self.feat_decoder(xs)

        if not self.cfg.EVAL.NO_LIFTING:
            # Depth distribution
            depth = self.depth(self.depth_decoder(xs)).softmax(dim=1)

            if self.sparse_depth:
                # only lift depth for topk most likely depth bins
                topk_bins = depth.topk(self.sparse_depth_count, dim=1)[1]
                depth_mask = torch.zeros(depth.shape, device=depth.device, dtype=torch.bool)
                depth_mask.scatter_(1, topk_bins, 1)
            else:
                depth_mask = torch.zeros(0, device=depth.device)
            x = (depth.unsqueeze(1) * x.unsqueeze(2)).type_as(x)  # outer product

            #  Add camera dimension
            x = x.unsqueeze(1)
            x = x.permute(0, 1, 3, 4, 5, 2)

            x = self.frustum_pooling(x, intrinsics.unsqueeze(1), extrinsics.unsqueeze(1), depth_mask)

        if self.cfg.MODEL.ROUTE.ENABLED:
            route_map = pack_sequence_dim(batch['route_map'])
            route_map_features = self.backbone_route(route_map)
            route_map_features = route_map_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, route_map_features], dim=1)

        if self.cfg.MODEL.MEASUREMENTS.ENABLED:
            route_command = pack_sequence_dim(batch['route_command'])
            gps_vector = pack_sequence_dim(batch['gps_vector'])
            route_command_next = pack_sequence_dim(batch['route_command_next'])
            gps_vector_next = pack_sequence_dim(batch['gps_vector_next'])

            command_features = self.command_encoder(route_command)
            command_features = command_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, command_features], dim=1)

            command_next_features = self.command_next_encoder(route_command_next)
            command_next_features = command_next_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, command_next_features], dim=1)

            gps_features = self.gps_encoder(torch.cat([gps_vector, gps_vector_next], dim=-1))
            gps_features = gps_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, gps_features], dim=1)

        speed_features = self.speed_enc(speed / self.speed_normalisation)
        speed_features = speed_features.unsqueeze(2).unsqueeze(3).expand(-1, -1, x.shape[2], x.shape[3])
        x = torch.cat((x, speed_features), 1)

        embedding = self.backbone_bev(x)[-1]
        embedding = self.final_state_conv(embedding)
        embedding = unpack_sequence_dim(embedding, b, s)
        return embedding

    def observe_and_imagine(self, batch, predict_action=False, future_horizon=None):
        """ This is only used for visualisation of future prediction"""
        assert self.cfg.MODEL.TRANSITION.ENABLED and self.cfg.SEMANTIC_SEG.ENABLED
        if future_horizon is None:
            future_horizon = self.cfg.FUTURE_HORIZON

        b, s = batch['image'].shape[:2]

        if not predict_action:
            assert batch['throttle_brake'].shape[1] == s + future_horizon
            assert batch['steering'].shape[1] == s + future_horizon

        # Observe past context
        output_observe = self.forward(batch)

        # Imagine future states
        output_imagine = {
            'action': [],
            'state': [],
            'hidden': [],
            'sample': [],
        }
        h_t = output_observe['posterior']['hidden_state'][:, -1]
        sample_t = output_observe['posterior']['sample'][:, -1]
        for t in range(future_horizon):
            if predict_action:
                action_t = self.policy(torch.cat([h_t, sample_t], dim=-1))
            else:
                action_t = torch.cat([batch['throttle_brake'][:, s+t], batch['steering'][:, s+t]], dim=-1)
            prior_t = self.rssm.imagine_step(
                h_t, sample_t, action_t, use_sample=True, policy=self.policy,
            )
            sample_t = prior_t['sample']
            h_t = prior_t['hidden_state']
            output_imagine['action'].append(action_t)
            output_imagine['state'].append(torch.cat([h_t, sample_t], dim=-1))
            output_imagine['hidden'].append(h_t)
            output_imagine['sample'].append(sample_t)

        for k, v in output_imagine.items():
            output_imagine[k] = torch.stack(v, dim=1)

        bev_decoder_output = self.bev_decoder(pack_sequence_dim(output_imagine['state']))
        bev_decoder_output = unpack_sequence_dim(bev_decoder_output, b, future_horizon)
        output_imagine = {**output_imagine, **bev_decoder_output}

        return output_observe, output_imagine

    def imagine(self, batch, predict_action=False, future_horizon=None):
        """ This is only used for visualisation of future prediction"""
        assert self.cfg.MODEL.TRANSITION.ENABLED and self.cfg.SEMANTIC_SEG.ENABLED
        if future_horizon is None:
            future_horizon = self.cfg.FUTURE_HORIZON

        # Imagine future states
        output_imagine = {
            'action': [],
            'state': [],
            'hidden': [],
            'sample': [],
        }
        h_t = batch['hidden_state'] #(b, c)
        sample_t = batch['sample']  #(b, s)
        b = h_t.shape[0]
        for t in range(future_horizon):
            if predict_action:
                action_t = self.policy(torch.cat([h_t, sample_t], dim=-1))
            else:
                action_t = torch.cat([batch['throttle_brake'][:, t], batch['steering'][:, t]], dim=-1)
            prior_t = self.rssm.imagine_step(
                h_t, sample_t, action_t, use_sample=True, policy=self.policy,
            )
            sample_t = prior_t['sample']
            h_t = prior_t['hidden_state']
            output_imagine['action'].append(action_t)
            output_imagine['state'].append(torch.cat([h_t, sample_t], dim=-1))
            output_imagine['hidden'].append(h_t)
            output_imagine['sample'].append(sample_t)

        for k, v in output_imagine.items():
            output_imagine[k] = torch.stack(v, dim=1)

        bev_decoder_output = self.bev_decoder(pack_sequence_dim(output_imagine['state']))
        bev_decoder_output = unpack_sequence_dim(bev_decoder_output, b, future_horizon)
        output_imagine = {**output_imagine, **bev_decoder_output}

        return output_imagine

    def deployment_forward(self, batch, is_dreaming):
        """
        Keep latent states in memory for fast inference.

        Parameters
        ----------
            batch: dict of torch.Tensor
                keys:
                    image: (b, s, 3, h, w)
                    route_map: (b, s, 3, h_r, w_r)
                    speed: (b, s, 1)
                    intrinsics: (b, s, 3, 3)
                    extrinsics: (b, s, 4, 4)
                    throttle_brake: (b, s, 1)
                    steering: (b, s, 1)
        """
        assert self.cfg.MODEL.TRANSITION.ENABLED
        b = batch['image'].shape[0]

        if self.count == 0:
            # Encode RGB images, route_map, speed using intrinsics and extrinsics
            # to a 512 dimensional vector
            s = batch['image'].shape[1]
            action_t = batch['action'][:, -2]  # action from t-1 to t
            batch = remove_past(batch, s)
            embedding_t = self.encode(batch)[:, -1]  # dim (b, 1, 512)

            # Recurrent state sequence module
            if self.last_h is None:
                h_t = action_t.new_zeros(b, self.cfg.MODEL.TRANSITION.HIDDEN_STATE_DIM)
                sample_t = action_t.new_zeros(b, self.cfg.MODEL.TRANSITION.STATE_DIM)
            else:
                h_t = self.last_h
                sample_t = self.last_sample

            if is_dreaming:
                rssm_output = self.rssm.imagine_step(
                    h_t, sample_t, action_t, use_sample=False, policy=self.policy,
                )
            else:
                rssm_output = self.rssm.observe_step(
                    h_t, sample_t, action_t, embedding_t, use_sample=False, policy=self.policy,
                )['posterior']
            sample_t = rssm_output['sample']
            h_t = rssm_output['hidden_state']

            self.last_h = h_t
            self.last_sample = sample_t

            game_frequency = CARLA_FPS
            model_stride_sec = self.cfg.DATASET.STRIDE_SEC
            n_image_per_stride = int(game_frequency * model_stride_sec)
            self.count = n_image_per_stride - 1
        else:
            self.count -= 1
        s = 1
        state = torch.cat([self.last_h, self.last_sample], dim=-1)
        output_policy = self.policy(state)
        throttle_brake, steering = torch.split(output_policy, 1, dim=-1)
        output = dict()
        output['throttle_brake'] = unpack_sequence_dim(throttle_brake, b, s)
        output['steering'] = unpack_sequence_dim(steering, b, s)

        output['hidden_state'] = self.last_h
        output['sample'] = self.last_sample

        if self.cfg.SEMANTIC_SEG.ENABLED and DISPLAY_SEGMENTATION:
            bev_decoder_output = self.bev_decoder(state)
            bev_decoder_output = unpack_sequence_dim(bev_decoder_output, b, s)
            output = {**output, **bev_decoder_output}

        return output
