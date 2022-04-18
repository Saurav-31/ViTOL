"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import argparse
import munch
import importlib
import os
from os.path import join as ospj
import shutil
import warnings

from util import Logger

_DATASET_NAMES = ('CUB', 'ILSVRC', 'OpenImages')
_ARCHITECTURE_NAMES = ('vgg16', 'resnet50', 'inception_v3', 'xception', 'vit', 'vitbase', 'vit_lrp')
_METHOD_NAMES = ('cam', 'adl', 'acol', 'spg', 'has', 'cutmix', 'rollout', 'vitol')
_SPLITS = ('train', 'val', 'test')


def mch(**kwargs):
    return munch.Munch(dict(**kwargs))


def box_v2_metric(args):
    if args.box_v2_metric:
        args.multi_contour_eval = True
        args.multi_iou_eval = True
    else:
        args.multi_contour_eval = False
        args.multi_iou_eval = False
        warnings.warn("MaxBoxAcc metric is deprecated.")
        warnings.warn("Use MaxBoxAccV2 by setting args.box_v2_metric to True.")


def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def configure_data_paths(args):
    train = val = test = ospj(args.data_root, args.dataset_name)
    data_paths = mch(train=train, val=val, test=test)
    return data_paths


def configure_mask_root(args):
    mask_root = ospj(args.mask_root, 'OpenImages')
    return mask_root


def configure_scoremap_output_paths(args):
    scoremaps_root = ospj(args.log_folder, 'scoremaps')
    scoremaps = mch()
    for split in ('train', 'val', 'test'):
        scoremaps[split] = ospj(scoremaps_root, split)
        if not os.path.isdir(scoremaps[split]):
            os.makedirs(scoremaps[split])
    return scoremaps


def configure_log_folder(args):
    log_folder = ospj('train_log', args.experiment_name)

    if os.path.isdir(log_folder):
        if args.override_cache:
            shutil.rmtree(log_folder, ignore_errors=True)
        else:
            raise RuntimeError("Experiment with the same name exists: {}"
                               .format(log_folder))
    os.makedirs(log_folder)
    return log_folder


def configure_log(args):
    log_file_name = ospj(args.log_folder, 'log.log')
    Logger(log_file_name)


def configure_reporter(args):
    reporter = importlib.import_module('util').Reporter
    reporter_log_root = ospj(args.log_folder, 'reports')
    if not os.path.isdir(reporter_log_root):
        os.makedirs(reporter_log_root)
    return reporter, reporter_log_root


def configure_pretrained_path(args):
    pretrained_path = None
    return pretrained_path


def check_dependency(args):
    if args.dataset_name == 'CUB':
        if args.num_val_sample_per_class >= 6:
            raise ValueError("num-val-sample must be <= 5 for CUB.")
    if args.dataset_name == 'OpenImages':
        if args.num_val_sample_per_class >= 26:
            raise ValueError("num-val-sample must be <= 25 for OpenImages.")

def get_configs():
    parser = argparse.ArgumentParser()

    # Util
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--experiment_name', type=str, default='test_case')
    parser.add_argument('--override_cache', type=str2bool, nargs='?',
                        const=True, default=False)
    parser.add_argument('--workers', default=1, type=int,
                        help='number of data loading workers (default: 4)')

    # Data
    parser.add_argument('--dataset_name', type=str, default='CUB',
                        choices=_DATASET_NAMES)
    parser.add_argument('--data_root', metavar='/PATH/TO/DATASET',
                        default='dataset/',
                        help='path to dataset images')
    parser.add_argument('--metadata_root', type=str, default='metadata/')
    parser.add_argument('--mask_root', metavar='/PATH/TO/MASKS',
                        default='dataset/',
                        help='path to masks')
    parser.add_argument('--proxy_training_set', type=str2bool, nargs='?',
                        const=True, default=False,
                        help='Efficient hyper_parameter search with a proxy '
                             'training set.')
    parser.add_argument('--num_val_sample_per_class', type=int, default=0,
                        help='Number of full_supervision validation sample per '
                             'class. 0 means "use all available samples".')

    # Setting
    parser.add_argument('--architecture', default='vit',
                        choices=_ARCHITECTURE_NAMES,
                        help='model architecture: ' +
                             ' | '.join(_ARCHITECTURE_NAMES) +
                             ' (default: vit)')
    parser.add_argument('--epochs', default=1, type=int,
                        help='number of total epochs to run')
    parser.add_argument('--pretrained', type=str2bool, nargs='?',
                        const=True, default=True,
                        help='Use pre_trained model.')
    parser.add_argument('--cam_curve_interval', type=float, default=.001,
                        help='CAM curve interval')
    parser.add_argument('--resize_size', type=int, default=256,
                        help='input resize size')
    parser.add_argument('--crop_size', type=int, default=224,
                        help='input crop size')
    parser.add_argument('--multi_contour_eval', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--multi_iou_eval', type=str2bool, nargs='?',
                        const=True, default=True)
    parser.add_argument('--iou_threshold_list', nargs='+',
                        type=int, default=[30, 50, 70])
    parser.add_argument('--eval_checkpoint_type', type=str, default='last',
                        choices=('best', 'last'))
    parser.add_argument('--box_v2_metric', type=str2bool, nargs='?',
                        const=True, default=True)

    # Common hyperparameters
    parser.add_argument('--batch_size', default=64, type=int,
                        help='Mini-batch size (default: 256), this is the total'
                             'batch size of all GPUs on the current node when'
                             'using Data Parallel or Distributed Data Parallel')

    parser.add_argument('--large_feature_map', type=str2bool, nargs='?',
                        const=True, default=False)

    # Method-specific hyperparameters
    parser.add_argument('--wsol_method', type=str, default='vitol',
                        choices=_METHOD_NAMES)
    parser.add_argument('--adl_drop_rate', type=float, default=0.75,
                        help='ADL dropout rate')
    parser.add_argument('--adl_threshold', type=float, default=0.9,
                        help='ADL gamma, threshold ratio '
                             'to maximum value of attention map')
    parser.add_argument('--evaluate_mode', type=str2bool, default=False,
                        help='Evaluate a model to generate scoremaps using last checkpoint')
    parser.add_argument('--base_model_dir', type=str, default="pretrained_weights",
                        help='Base model to use for generating scoremaps')
    parser.add_argument('--adl_layer', type=str2bool, default=False,
                        help='Whether to use ADL inside transformer')
    parser.add_argument('--vit_type', type=str, default='vit',
                        help='vit type: vit, vit_deit, vit_large')
    parser.add_argument('--eval_method', type=str, default='lrp',
                        help='Please Choos new: lrp or baseline method: rollout or new method grad_rollout')
    parser.add_argument('--scoremap_threshold', type=float, default=0.9,
                        help='scoremap_threshold '
                             'scoremap_threshold')
    parser.add_argument('--ckpt_name', type=str, default='ViTOL-DeiT-B_IMAGENET_last.pth.tar',
                        help='Name of the checkpoint model')
                             
    args = parser.parse_args()

    check_dependency(args)
    args.log_folder = configure_log_folder(args)
    configure_log(args)
    box_v2_metric(args)

    args.architecture_type = args.wsol_method
    args.data_paths = configure_data_paths(args)
    args.metadata_root = ospj(args.metadata_root, args.dataset_name)
    args.mask_root = configure_mask_root(args)
    args.scoremap_paths = configure_scoremap_output_paths(args)
    args.reporter, args.reporter_log_root = configure_reporter(args)
    args.pretrained_path = configure_pretrained_path(args)
    
    return args
