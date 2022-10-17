""" Adapted from https://github.com/nv-tlabs/lift-splat-shoot/blob/master/src/tools.py"""

import numpy as np
import torch
import torch.nn as nn

from mile.utils.geometry_utils import bev_params_to_intrinsics, intrinsics_inverse


def gen_dx_bx(size, scale, offsetx):
    xbound = [-size[0] * scale / 2 - offsetx * scale, size[0] * scale / 2 - offsetx * scale, scale]
    ybound = [-size[1] * scale / 2, size[1] * scale / 2, scale]
    zbound = [-10.0, 10.0, 20.0]

    dx = torch.Tensor([row[2] for row in [xbound, ybound, zbound]])
    bx = torch.Tensor([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    # nx = torch.LongTensor([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    nx = torch.LongTensor([np.round((row[1] - row[0]) / row[2]) for row in [xbound, ybound, zbound]])

    return dx, bx, nx


def cumsum_trick(x, geom_feats, ranks):
    x = x.cumsum(0)
    kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
    kept[:-1] = (ranks[1:] != ranks[:-1])

    x, geom_feats = x[kept], geom_feats[kept]
    x = torch.cat((x[:1], x[1:] - x[:-1]))

    return x, geom_feats


class QuickCumsum(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, geom_feats, ranks):
        x = x.cumsum(0)
        kept = torch.ones(x.shape[0], device=x.device, dtype=torch.bool)
        kept[:-1] = (ranks[1:] != ranks[:-1])

        x, geom_feats = x[kept], geom_feats[kept]
        x = torch.cat((x[:1], x[1:] - x[:-1]))

        # save kept for backward
        ctx.save_for_backward(kept)

        # no gradient for geom_feats
        ctx.mark_non_differentiable(geom_feats)

        return x, geom_feats

    @staticmethod
    def backward(ctx, gradx, gradgeom):
        kept, = ctx.saved_tensors
        back = torch.cumsum(kept, 0)
        back[kept] -= 1

        val = gradx[back]

        return val, None, None


def quick_cumsum(x, geom_feats, ranks):
    return QuickCumsum.apply(x, geom_feats, ranks)


class FrustumPooling(nn.Module):
    def __init__(self, size, scale, offsetx, dbound, downsample, use_quickcumsum=True):
        """ Pools camera frustums into Birds Eye View

        Args:
            size: (width, height) size of voxel grid
            scale: size of pixel in m
            offsetx: egocar offset (forwards) from center of bev in px
            dbound: depth planes in camera frustum (min, max, step)
            downsample: fraction of the size of the feature maps (stride of backbone)
        """
        super().__init__()

        self.register_buffer('bev_intrinsics', torch.tensor(bev_params_to_intrinsics(size, scale, offsetx)))

        dx, bx, nx = gen_dx_bx(size, scale, offsetx)
        self.nx_constant = nx.numpy().tolist()
        self.register_buffer('dx', dx, persistent=False)  # bev_resolution
        self.register_buffer('bx', bx, persistent=False)  # bev_start_position
        self.register_buffer('nx', nx, persistent=False)  # bev_dimension
        self.use_quickcumsum = use_quickcumsum

        self.dbound = dbound
        ds = torch.arange(self.dbound[0], self.dbound[1], self.dbound[2], dtype=torch.float32)
        self.D = len(ds)
        self.register_buffer('ds', ds, persistent=False)

        self.downsample = downsample
        self.register_buffer('frustum', torch.zeros(0,), persistent=False)

    def initialize_frustum(self, image):
        if self.frustum.shape[0] == 0:
            device = image.device
            # make grid in image plane
            fH, fW = image.shape[-3:-1]
            ogfH, ogfW = fH * self.downsample, fW * self.downsample
            ds = self.ds.view(-1, 1, 1).expand(-1, fH, fW)
            xs = torch.linspace(0, ogfW - 1, fW, dtype=torch.float, device=device).view(1, 1, fW).expand(self.D, fH, fW)
            ys = torch.linspace(0, ogfH - 1, fH, dtype=torch.float, device=device).view(1, fH, 1).expand(self.D, fH, fW)

            # D x H x W x 3
            # with the 3D coordinates being (x, y, z)
            self.frustum = torch.stack((xs, ys, ds), -1)

    def get_geometry(self, rots, trans, intrins):  # , post_rots=None, post_trans=None):
        """Determine the (x,y,z) locations (in the ego frame)
        of the points in the point cloud.
        Returns B x N x D x H/downsample x W/downsample x 3
        """
        B, N = trans.shape[:2]

        points = self.frustum.unsqueeze(0).unsqueeze(0).unsqueeze(-1)

        # cam_to_ego
        points = torch.cat((points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                            points[:, :, :, :, :, 2:3]
                            ), 5)
        # combine = rots.matmul(torch.inverse(intrins))
        combine = rots.matmul(intrinsics_inverse(intrins))
        points = combine.view(B, N, 1, 1, 1, 3, 3).matmul(points).squeeze(-1)
        points += trans.view(B, N, 1, 1, 1, 3)

        return points

    def voxel_pooling(self, geom_feats, x, mask):
        B, N, D, H, W, C = x.shape
        Nprime = B * N * D * H * W

        # flatten
        x = x.reshape(Nprime, C)

        # The coordinates are defined as (forward, left, up)
        geom_feats = geom_feats.view(Nprime, 3)

        # transform world points to bev coords
        geom_feats[:, 0] = geom_feats[:, 0] * self.bev_intrinsics[0, 0] + self.bev_intrinsics[0, 2]
        geom_feats[:, 1] = geom_feats[:, 1] * self.bev_intrinsics[1, 1] + self.bev_intrinsics[1, 2]
        # TODO: seems like things < -10m also get projected.
        geom_feats[:, 2] = (geom_feats[:, 2] - self.bx[2] + self.dx[2] / 2.) / self.dx[2]
        geom_feats = geom_feats.long()

        batch_ix = torch.cat([torch.full(size=(Nprime // B, 1), fill_value=ix,
                                         device=x.device, dtype=torch.long) for ix in range(B)])
        geom_feats = torch.cat((geom_feats, batch_ix), 1)

        # sparse lifting for speed
        if len(mask) > 0:
            mask = mask.view(Nprime)
            x = x[mask]
            geom_feats = geom_feats[mask]

        # filter out points that are outside box
        kept = (geom_feats[:, 0] >= 0) & (geom_feats[:, 0] < self.nx[0]) \
               & (geom_feats[:, 1] >= 0) & (geom_feats[:, 1] < self.nx[1]) \
               & (geom_feats[:, 2] >= 0) & (geom_feats[:, 2] < self.nx[2])
        x = x[kept]
        geom_feats = geom_feats[kept]

        # get tensors from the same voxel next to each other
        ranks = geom_feats[:, 0] * (self.nx[1] * self.nx[2] * B) \
                + geom_feats[:, 1] * (self.nx[2] * B) \
                + geom_feats[:, 2] * B \
                + geom_feats[:, 3]
        sorts = ranks.argsort()
        x, geom_feats, ranks = x[sorts], geom_feats[sorts], ranks[sorts]

        # cumsum trick
        if self.use_quickcumsum and self.training:
            x, geom_feats = quick_cumsum(x, geom_feats, ranks)
        else:
            x, geom_feats = cumsum_trick(x, geom_feats, ranks)

        # griddify (B x C x up x left x forward)
        final = torch.zeros((B, C, self.nx_constant[2], self.nx_constant[1], self.nx_constant[0]), dtype=x.dtype,
                            device=x.device)
        final[geom_feats[:, 3], :, geom_feats[:, 2], geom_feats[:, 1], geom_feats[:, 0]] = x

        # collapse "up" dimension
        final = torch.cat(final.unbind(dim=2), 1)

        return final

    def forward(self, x, intrinsics, pose, mask=torch.zeros(0)):  # , post_rots=None, post_trans=None):
        """
        Args:
            x: (B x N x D x H x W x C) frustum feature maps
            intrinsics: (B x N x 3 x 3) camera intrinsics (of input image prior to downsampling by backbone)
            pose: (B x N x 4 x 4) camera pose matrix
        """

        # the intrinsics matrix is defined as
        # [[f', 0, m_x],
        #  [0, f', m_y],
        #  [0, 0, 1]]
        # with f' = kf in pixel units. k being the factor in pixel/m, f the focal lens in m.
        # (m_x, m_y) is the center point in pixel.

        self.initialize_frustum(x)
        rots = pose[..., :3, :3]
        trans = pose[..., :3, 3:]
        geom = self.get_geometry(rots, trans, intrinsics)  # , post_rots, post_trans)
        x = self.voxel_pooling(geom, x, mask).type_as(x)  # TODO: do we want to do more of frustum pooling in FP16?
        return x

    def get_depth_map(self, depth):
        """ Convert depth probibility distribution to depth """
        ds = self.ds.view(1, -1, 1, 1)
        depth = (ds * depth).sum(1, keepdim=True)
        depth = nn.functional.interpolate(depth, scale_factor=float(self.downsample), mode='bilinear',
                                          align_corners=False)
        return depth
