"""Adapted from https://github.com/zhejz/carla-roach CC-BY-NC 4.0 license."""
import numpy as np
import torch
import carla

from mile.visualisation import prepare_final_display_image, convert_bev_to_image, upsample_bev


class MileWrapper:
    def __init__(self, cfg=None):
        self.cfg = cfg

    def process_act(self, action):
        acc, steer = action.astype(np.float64)
        if acc >= 0.0:
            throttle = acc
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.abs(acc)

        throttle = np.clip(throttle, 0, 1)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)
        control = carla.VehicleControl(throttle=throttle, steer=steer, brake=brake)
        return control

    def im_render(self, render_dict, upsample_bev_factor=2):
        im_rgb = render_dict['policy_input']['image'][0, -1].cpu().numpy().transpose((1, 2, 0))
        route_map = render_dict['policy_input']['route_map'][0, -1].cpu().numpy().transpose((1, 2, 0))

        # Un-normalise images
        img_mean = np.array(render_dict['policy_cfg'].IMAGE.IMAGENET_MEAN)
        img_std = np.array(render_dict['policy_cfg'].IMAGE.IMAGENET_STD)
        im_rgb = (255 * (im_rgb * img_std + img_mean)).astype(np.uint8)
        route_map = (255 * (route_map * img_std + img_mean)).astype(np.uint8)

        birdview_label = render_dict['policy_input']['birdview_label'][0, -1]
        birdview_label = torch.rot90(birdview_label, k=1, dims=[1, 2])
        birdview_label = upsample_bev(birdview_label)
        # Add colours
        policy_cfg = render_dict['policy_cfg']
        birdview_label_rendered = convert_bev_to_image(
            birdview_label.cpu().numpy()[0], cfg=policy_cfg, upsample_factor=upsample_bev_factor,
        )

        # Check if the bev predictions are in the outputs
        if 'bev_segmentation_1' in render_dict:
            bev_prediction = render_dict['bev_segmentation_1'][0, -1]
            # Rotate prediction
            bev_prediction = torch.rot90(bev_prediction, k=1, dims=[1, 2])
            bev_prediction = torch.argmax(bev_prediction, dim=0, keepdim=True)
            bev_prediction = upsample_bev(bev_prediction)
            bev_prediction = convert_bev_to_image(
                bev_prediction.cpu().numpy()[0], policy_cfg, upsample_factor=upsample_bev_factor,
            )

        final_display_image = prepare_final_display_image(
            im_rgb, route_map, birdview_label_rendered, bev_prediction, render_dict
        )

        return final_display_image

