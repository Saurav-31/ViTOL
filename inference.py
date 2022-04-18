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

import cv2
import numpy as np
import os
from os.path import join as ospj
from os.path import dirname as ospd

from evaluation import BoxEvaluator
from evaluation import configure_metadata
from util import t2n
from tqdm import tqdm
import torch
from wsol_model.vitol import generate_cam
from explainability.ViT_explanation_generator import LRP, Baselines
import matplotlib.pyplot as plt

# _IMAGENET_MEAN = [0.485, .456, .406]
# _IMAGENET_STDDEV = [.229, .224, .225]

_IMAGENET_MEAN = [0.5, 0.5, 0.5]
_IMAGENET_STDDEV = [0.5, 0.5, 0.5]
_RESIZE_LENGTH = 224

def normalize_scoremap(cam):
    """
    Args:
        cam: numpy.ndarray(size=(H, W), dtype=np.float)
    Returns:
        numpy.ndarray(size=(H, W), dtype=np.float) between 0 and 1.
        If input array is constant, a zero-array is returned.
    """
    if np.isnan(cam).any():
        return np.zeros_like(cam)
    if cam.min() == cam.max():
        return np.zeros_like(cam)
    cam -= cam.min()
    cam /= cam.max()
    return cam


class CAMComputer(object):
    def __init__(self, model, loader, metadata_root, mask_root,
                 iou_threshold_list, dataset_name, split,
                 multi_contour_eval, args, cam_curve_interval=.001, log_folder=None, split_arg= False):
        self.model = model
        self.model.eval()
        self.loader = loader
        self.split = split
        self.log_folder = log_folder
        self.args = args

        self.lrp = False
        self.rollout = False
        self.grad_rollout = False

        if self.args.architecture_type == 'vitol' and self.args.eval_method == 'lrp':
            self.lrp = True
        elif self.args.architecture_type == 'vitol' and self.args.eval_method == 'rollout':
            self.rollout = True
        elif self.args.architecture_type == 'vitol' and self.args.eval_method == 'grad_rollout':
            self.grad_rollout = True

        metadata = configure_metadata(metadata_root)
        # cam_threshold_list = list(np.arange(0, 1, cam_curve_interval))
        cam_threshold_list = [self.args.scoremap_threshold]
        print("Evaluation is done at fixed threshold {}".format(self.args.scoremap_threshold))
        self.evaluator = {"CUB": BoxEvaluator,
                          "ILSVRC": BoxEvaluator
                          }[dataset_name](metadata=metadata,
                                          dataset_name=dataset_name,
                                          split=split,
                                          cam_threshold_list=cam_threshold_list,
                                          iou_threshold_list=iou_threshold_list,
                                          mask_root=mask_root,
                                          multi_contour_eval=multi_contour_eval)

    def compute_and_evaluate_cams(self):
        print("Computing and evaluating cams.")

        if self.lrp:
            attribution_generator = LRP(self.model)
        elif self.rollout or self.grad_rollout:
            attribution_generator = Baselines(self.model)

        num_images = 0
        top1_loc = 0.0

        for images, targets, image_ids in tqdm(self.loader):
            image_size = images.shape[2:]
            images = images.cuda()
            # images = images.to(device)

            num_images += images.size(0)

            if self.lrp or self.rollout or self.grad_rollout:
                out = self.model(images)
                cams = generate_cam(attribution_generator, images, class_index=targets, eval_method = self.args.eval_method)
            else:
                out = self.model(images)
                cams = self.model(images, targets, return_cam=True)

            if self.args.vit_type == 'vit_deit' or self.args.architecture_type == 'vitol':
                pred = out.argmax(dim=1)
            else:
                pred = out['logits'].argmax(dim=1)

            if torch.is_tensor(cams):
                cams = t2n(cams)

            split_list = ('val', 'test')
            i = 0
            for cam, image_id in zip(cams, image_ids):
                cam_resized = cv2.resize(cam, image_size,
                                         interpolation=cv2.INTER_CUBIC)
                cam_normalized = normalize_scoremap(cam_resized)
                
                if self.split in split_list:
                    cam_path = ospj(self.log_folder, 'scoremaps', image_id)
                    if not os.path.exists(ospd(cam_path)):
                        os.makedirs(ospd(cam_path))
                    np.save(ospj(cam_path), cam_normalized)
                    plt.imsave(ospj(cam_path), cam_normalized)
                # self.evaluator.accumulate(cam_normalized, image_id)

                classification_flag = (int(pred[i]) == int(targets[i]))
                i+=1
                self.evaluator.accumulate(cam_normalized, image_id, classification_flag)

        max_box_acc, top1_loc_acc = self.evaluator.compute()
        print("Top1 Localization acc: ", top1_loc_acc)

        return max_box_acc, top1_loc_acc
