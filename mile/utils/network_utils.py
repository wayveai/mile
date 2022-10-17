import torch
import torch.nn as nn
import torchvision


def set_bn_momentum(model, momentum=0.1):
    for m in model.modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            m.momentum = momentum


def preprocess_batch(batch, device, unsqueeze=False):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.to(device)
            if unsqueeze:
                batch[key] = batch[key].unsqueeze(0)
        else:
            preprocess_batch(value, device, unsqueeze=unsqueeze)


def squeeze_batch(batch):
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            batch[key] = value.squeeze(0)
        else:
            squeeze_batch(value)


def remove_past(x, receptive_field):
    """ Removes past tensors. The past is indicated by the receptive field. Creates a copy."""
    if isinstance(x, torch.Tensor):
        return x[:, (receptive_field-1):].contiguous()

    output = {}
    for key, value in x.items():
        output[key] = remove_past(value, receptive_field)
    return output


def remove_last(x):
    """ Removes last timestep. Creates a copy."""
    if isinstance(x, torch.Tensor):
        return x[:, :-1].contiguous()

    output = {}
    for key, value in x.items():
        output[key] = remove_last(value)
    return output


def pack_sequence_dim(x):
    """ Does not create a copy."""
    if isinstance(x, torch.Tensor):
        b, s = x.shape[:2]
        return x.view(b * s, *x.shape[2:])

    if isinstance(x, list):
        return [pack_sequence_dim(elt) for elt in x]

    output = {}
    for key, value in x.items():
        output[key] = pack_sequence_dim(value)
    return output


def unpack_sequence_dim(x, b, s):
    """ Does not create a copy."""
    if isinstance(x, torch.Tensor):
        return x.view(b, s, *x.shape[1:])

    if isinstance(x, list):
        return [unpack_sequence_dim(elt, b, s) for elt in x]

    output = {}
    for key, value in x.items():
        output[key] = unpack_sequence_dim(value, b, s)
    return output


def select_time_indices(x, time_indices):
    """
    Selects a particular time index for each element in the batch. Creates a copy.

    Parameters
    ----------
        x: dict of tensors shape (batch_size, sequence_length, ...)
        time_indices: torch.int64 shape (batch_size)
    """
    if isinstance(x, torch.Tensor):
        b = x.shape[0]
        return x[torch.arange(b), time_indices]

    if isinstance(x, list):
        return [select_time_indices(elt, time_indices) for elt in x]

    output = {}
    for key, value in x.items():
        output[key] = select_time_indices(value, time_indices)
    return output


def calculate_birds_eye_view_parameters(x_bounds, y_bounds, z_bounds):
    """
    Parameters
    ----------
        x_bounds: Forward direction in the ego-car.
        y_bounds: Sides
        z_bounds: Height
    Returns
    -------
        bev_resolution: Bird's-eye view bev_resolution
        bev_start_position Bird's-eye view first element
        bev_dimension Bird's-eye view tensor spatial dimension
    """
    bev_resolution = torch.tensor([row[2] for row in [x_bounds, y_bounds, z_bounds]])
    bev_start_position = torch.tensor([row[0] + row[2] / 2.0 for row in [x_bounds, y_bounds, z_bounds]])
    bev_dimension = torch.tensor([(row[1] - row[0]) / row[2] for row in [x_bounds, y_bounds, z_bounds]],
                                 dtype=torch.long)

    return bev_resolution, bev_start_position, bev_dimension


class NormalizeInverse(torchvision.transforms.Normalize):
    #  https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821/8
    def __init__(self, mean, std):
        mean = torch.as_tensor(mean)
        std = torch.as_tensor(std)
        std_inv = 1 / (std + 1e-7)
        mean_inv = -mean * std_inv
        super().__init__(mean=mean_inv, std=std_inv)

    def __call__(self, tensor):
        return super().__call__(tensor.clone())


def freeze_network(network):
    for p in network.parameters():
        p.requires_grad = False


def unfreeze_network(network):
    for p in network.parameters():
        p.requires_grad = True
