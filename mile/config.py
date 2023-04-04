import argparse
from fvcore.common.config import CfgNode as _CfgNode


def convert_to_dict(cfg_node, key_list=[]):
    """Convert a config node to dictionary."""
    _VALID_TYPES = {tuple, list, str, int, float, bool}
    if not isinstance(cfg_node, _CfgNode):
        if type(cfg_node) not in _VALID_TYPES:
            print(
                'Key {} with value {} is not a valid type; valid types: {}'.format(
                    '.'.join(key_list), type(cfg_node), _VALID_TYPES
                ),
            )
        return cfg_node
    else:
        cfg_dict = dict(cfg_node)
        for k, v in cfg_dict.items():
            cfg_dict[k] = convert_to_dict(v, key_list + [k])
        return cfg_dict


class CfgNode(_CfgNode):
    """Remove once https://github.com/rbgirshick/yacs/issues/19 is merged."""

    def convert_to_dict(self):
        return convert_to_dict(self)

CN = CfgNode

_C = CN()
_C.LOG_DIR = 'tensorboard_logs'
_C.TAG = 'default'

_C.GPUS = 1  # how many gpus to use
_C.PRECISION = 16  # 16bit or 32bit
_C.BATCHSIZE = 2
_C.STEPS = 50000
_C.N_WORKERS = 8

_C.VAL_CHECK_INTERVAL = 5000
_C.LOGGING_INTERVAL = 500
_C.LIMIT_VAL_BATCHES = 1

_C.RECEPTIVE_FIELD = 1
_C.FUTURE_HORIZON = 1

###########
# Optimizer
###########
_C.OPTIMIZER = CN()
_C.OPTIMIZER.LR = 1e-4
_C.OPTIMIZER.WEIGHT_DECAY = 0.01
_C.OPTIMIZER.ACCUMULATE_GRAD_BATCHES = 1

_C.SCHEDULER = CN()
_C.SCHEDULER.NAME = 'OneCycleLR'
_C.SCHEDULER.PCT_START = 0.2

#########
# Dataset
#########
_C.DATASET = CN()
_C.DATASET.DATAROOT = ''
_C.DATASET.VERSION = 'trainval'
_C.DATASET.STRIDE_SEC = 0.2  # stride between two frames
_C.DATASET.FILTER_BEGINNING_OF_RUN_SEC = 1.0  # in seconds. the beginning of the run is stationary.
_C.DATASET.FILTER_NORM_REWARD = 0.6  # filter runs that have a normalised reward below this value.


#############
# Input image
#############
_C.IMAGE = CN()
_C.IMAGE.SIZE = (600, 960)
_C.IMAGE.CROP = [64, 138, 896, 458]  # (left, top, right, bottom)
_C.IMAGE.FOV = 100
_C.IMAGE.CAMERA_POSITION = [-1.5, 0.0, 2.0]  # (forward, right, up)

# carla defines as (pitch, yaw, roll)
# /!\ roach defines as (roll, pitch, yaw)
# this is fine for now because all of them are equal to zero.
_C.IMAGE.CAMERA_ROTATION = [0.0, 0.0, 0.0]
_C.IMAGE.IMAGENET_MEAN = (0.485, 0.456, 0.406)
_C.IMAGE.IMAGENET_STD = (0.229, 0.224, 0.225)

_C.IMAGE.AUGMENTATION = CN()  # image augmentations
_C.IMAGE.AUGMENTATION.BLUR_PROB = .3
_C.IMAGE.AUGMENTATION.BLUR_WINDOW = 5
_C.IMAGE.AUGMENTATION.BLUR_STD = [.1, 1.7]
_C.IMAGE.AUGMENTATION.SHARPEN_PROB = .3
_C.IMAGE.AUGMENTATION.SHARPEN_FACTOR = [1, 5]
_C.IMAGE.AUGMENTATION.COLOR_PROB = .3
_C.IMAGE.AUGMENTATION.COLOR_JITTER_BRIGHTNESS = .3
_C.IMAGE.AUGMENTATION.COLOR_JITTER_CONTRAST = .3
_C.IMAGE.AUGMENTATION.COLOR_JITTER_SATURATION = .3
_C.IMAGE.AUGMENTATION.COLOR_JITTER_HUE = .1

_C.BEV = CN()
_C.BEV.SIZE = [192, 192]  # width, height. note that the bev is rotated, so width corresponds to forward direction.
_C.BEV.RESOLUTION = 0.2  # pixel size in m
_C.BEV.OFFSET_FORWARD = -64  # offset of the center of gravity of the egocar relative to the center of bev in px
_C.BEV.FEATURE_DOWNSAMPLE = 4  # Downsample factor for bev features

_C.BEV.FRUSTUM_POOL = CN()
_C.BEV.FRUSTUM_POOL.D_BOUND = [1.0, 38.0, 1.0]
_C.BEV.FRUSTUM_POOL.SPARSE = True
_C.BEV.FRUSTUM_POOL.SPARSE_COUNT = 10

###########
# Route map
###########
_C.ROUTE = CN()
_C.ROUTE.SIZE = 64  # spatial resolution

_C.ROUTE.AUGMENTATION_DROPOUT = .025
_C.ROUTE.AUGMENTATION_END_OF_ROUTE = .025
_C.ROUTE.AUGMENTATION_SMALL_ROTATION = .025
_C.ROUTE.AUGMENTATION_LARGE_ROTATION = .025
_C.ROUTE.AUGMENTATION_DEGREES = 8.
_C.ROUTE.AUGMENTATION_TRANSLATE = (.1, .1)
_C.ROUTE.AUGMENTATION_SCALE = (.95, 1.05)
_C.ROUTE.AUGMENTATION_SHEAR = (.1, .1)

#######
# Speed
#######
_C.SPEED = CN()
_C.SPEED.NOISE_STD = 1.4  # in m/s
_C.SPEED.NORMALISATION = 5.0  # in m/s

#######
# Model
#######
_C.MODEL = CN()

_C.MODEL.ACTION_DIM = 2

_C.MODEL.ENCODER = CN()
_C.MODEL.ENCODER.NAME = 'resnet18'
_C.MODEL.ENCODER.OUT_CHANNELS = 64

_C.MODEL.BEV = CN()
_C.MODEL.BEV.BACKBONE = 'resnet18'
_C.MODEL.BEV.CHANNELS = 64

_C.MODEL.SPEED = CN()
_C.MODEL.SPEED.CHANNELS = 16

_C.MODEL.ROUTE = CN()
_C.MODEL.ROUTE.ENABLED = True
_C.MODEL.ROUTE.BACKBONE = 'resnet18'
_C.MODEL.ROUTE.CHANNELS = 16

_C.MODEL.MEASUREMENTS = CN()
_C.MODEL.MEASUREMENTS.ENABLED = False
_C.MODEL.MEASUREMENTS.COMMAND_CHANNELS = 8
_C.MODEL.MEASUREMENTS.GPS_CHANNELS = 16

_C.MODEL.EMBEDDING_DIM = 512

_C.MODEL.TRANSITION = CN()
_C.MODEL.TRANSITION.ENABLED = True
_C.MODEL.TRANSITION.HIDDEN_STATE_DIM = 1024  # Dimention of the RNN hidden representation
_C.MODEL.TRANSITION.STATE_DIM = 512  # Dimension of prior/posterior
_C.MODEL.TRANSITION.ACTION_LATENT_DIM = 64  # Latent dimension of action
_C.MODEL.TRANSITION.USE_DROPOUT = True
_C.MODEL.TRANSITION.DROPOUT_PROBABILITY = 0.15

###########
# LOSSES
###########
_C.SEMANTIC_SEG = CN()
_C.SEMANTIC_SEG.ENABLED = True
_C.SEMANTIC_SEG.N_CHANNELS = 8
_C.SEMANTIC_SEG.USE_TOP_K = True  # backprop only top-k hardest pixels
_C.SEMANTIC_SEG.TOP_K_RATIO = 0.25
_C.SEMANTIC_SEG.USE_WEIGHTS = True

# Always enabled with seg
_C.INSTANCE_SEG = CN()
_C.INSTANCE_SEG.CENTER_LABEL_SIGMA_PX = 4
_C.INSTANCE_SEG.IGNORE_INDEX = 255
_C.INSTANCE_SEG.CENTER_LOSS_WEIGHT = 200.0
_C.INSTANCE_SEG.OFFSET_LOSS_WEIGHT = 0.1

_C.LOSSES = CN()
_C.LOSSES.WEIGHT_ACTION = 1.0
_C.LOSSES.WEIGHT_SEGMENTATION = 0.1
_C.LOSSES.WEIGHT_INSTANCE = 0.1
_C.LOSSES.WEIGHT_PROBABILISTIC = 1e-3
_C.LOSSES.KL_BALANCING_ALPHA = 0.75

_C.PRETRAINED = CN()
_C.PRETRAINED.PATH = ''

# There parameters are only used to benchmark other models.
_C.EVAL = CN()
_C.EVAL.RGB_SUPERVISION = False
_C.EVAL.NO_LIFTING = False
# Dataset size experiments
_C.EVAL.DATASET_REDUCTION = False
_C.EVAL.DATASET_REDUCTION_FACTOR = 1
# Image resolution experiments
_C.EVAL.RESOLUTION = CN()
_C.EVAL.RESOLUTION.ENABLED = False
_C.EVAL.RESOLUTION.FACTOR = 1

#########
# Sampler
#########
_C.SAMPLER = CN()
_C.SAMPLER.ENABLED = False
_C.SAMPLER.WITH_ACCELERATION = False
_C.SAMPLER.WITH_STEERING = False
_C.SAMPLER.N_BINS = 5
_C.SAMPLER.WITH_ROUTE_COMMAND = False  # not used
_C.SAMPLER.COMMAND_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0]


def get_parser():
    parser = argparse.ArgumentParser(description='World model training')
    parser.add_argument('--config-file', default='', metavar='FILE', help='path to config file')
    parser.add_argument(
        'opts', help='Modify config options using the command-line', default=None, nargs=argparse.REMAINDER,
    )
    return parser


def _find_extra_keys(dict1, dict2, path=''):
    """
    Recursively finds keys that exist in dict2 but not in dict1.
    Returns the full path of the missing keys, including the parent key names.
    """
    results = []
    for key in dict2.keys():
        new_path = f"{path}.{key}" if path else key
        if key in dict1:
            if isinstance(dict1[key], dict) and isinstance(dict2[key], dict):
                results.extend(_find_extra_keys(dict1[key], dict2[key], new_path))
        else:
            results.append(new_path)
        results.sort()
    return results


def get_cfg(args=None, cfg_dict=None):
    """First get default config. Then merge cfg_dict. Then merge according to args."""

    cfg = _C.clone()

    if cfg_dict is not None:
        extra_keys = _find_extra_keys(cfg, cfg_dict)
        if len(extra_keys) > 0:
            print(f"Warning - the cfg_dict merging into the main cfg has keys that do not exist in main: {extra_keys}")
            cfg.set_new_allowed(True)
        cfg.merge_from_other_cfg(CfgNode(cfg_dict))

    if args is not None:
        if args.config:
            cfg.merge_from_file(args.config)

        if args.config1:
            cfg.merge_from_file(args.config1)

        if args.opts:
            cfg.merge_from_list(args.opts)

    return cfg