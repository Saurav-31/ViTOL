import os
import pdb
import random

import matplotlib
import numpy as np
import torch
from tqdm import tqdm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wsol_model.vitol import vitol
from config import get_configs
from data_loaders import get_data_loader
from inference import CAMComputer

def set_random_seed(seed=123):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True 

class PerformanceMeter(object):
    def __init__(self, split, higher_is_better=True):
        self.best_function = max if higher_is_better else min
        self.current_value = None
        self.best_value = None
        self.best_epoch = None
        self.value_per_epoch = [] \
            if split == 'val' else [-np.inf if higher_is_better else np.inf]

    def update(self, new_value):
        self.value_per_epoch.append(new_value)
        self.current_value = self.value_per_epoch[-1]
        self.best_value = self.best_function(self.value_per_epoch)
        self.best_epoch = self.value_per_epoch.index(self.best_value)

class ViTOLInference():
    _NUM_CLASSES_MAPPING = {
        "CUB": 200,
        "ILSVRC": 1000,
    }
    _SPLITS = ('train', 'val', 'test')
    _CHECKPOINT_NAME_TEMPLATE = '{}_checkpoint.pth.tar'
    _EVAL_METRICS = ['loss', 'classification', 'localization', 'top1_loc_acc']
    _BEST_CRITERION_METRIC = 'top1_loc_acc'

    def __init__(self):
        self.args = get_configs()
        set_random_seed(self.args.seed)
        print("---------------------------------------------------------------------------------------------------")
        print(f'Dataset: {self.args.dataset_name}, Experiment: {self.args.experiment_name}, vit type: {self.args.vit_type}, maxboxv2: {self.args.box_v2_metric}')
        print(f'Adl_layer: {self.args.adl_layer}, adl_drop_rate: {self.args.adl_drop_rate}, adl_threshold: {self.args.adl_threshold}')
        print(f'Architecture: {self.args.architecture}, architecture_type: {self.args.architecture_type}, wsol_method: {self.args.wsol_method}')
        print(f'BS: {self.args.batch_size}, Epochs: {self.args.epochs}')
        print(f'Eval mode: {self.args.evaluate_mode}, Eval method: {self.args.eval_method}, eval ckpt type: {self.args.eval_checkpoint_type}, base model: {self.args.base_model_dir}')
        print("---------------------------------------------------------------------------------------------------")
        self.performance_meters = self._set_performance_meters()
        
        self.model = self._set_model()
        self.loaders = get_data_loader(
            data_roots=self.args.data_paths,
            metadata_root=self.args.metadata_root,
            batch_size=self.args.batch_size,
            workers=self.args.workers,
            resize_size=self.args.resize_size,
            crop_size=self.args.crop_size,
            proxy_training_set=self.args.proxy_training_set,
            num_val_sample_per_class=self.args.num_val_sample_per_class,
        )

    def _set_performance_meters(self):
        self._EVAL_METRICS += ['localization_IOU_{}'.format(threshold)
                               for threshold in self.args.iou_threshold_list]

        eval_dict = {
            split: {
                metric: PerformanceMeter(split,
                                            higher_is_better=False
                                            if metric == 'loss' else True)
                for metric in self._EVAL_METRICS
            }
            for split in self._SPLITS
        }
        return eval_dict

    def _set_model(self):
        num_classes = self._NUM_CLASSES_MAPPING[self.args.dataset_name]
        print("Loading model {}".format(self.args.architecture))
        model = vitol(
            dataset_name=self.args.dataset_name,
            architecture_type=self.args.architecture_type,
            pretrained=self.args.pretrained,
            num_classes=num_classes,
            large_feature_map=self.args.large_feature_map,
            pretrained_path=self.args.pretrained_path,
            adl_drop_rate=self.args.adl_drop_rate,
            adl_drop_threshold=self.args.adl_threshold,
            adl_layer = self.args.adl_layer,
            vit_type=self.args.vit_type,
        )
        model = model.cuda()
        print(model._modules['blocks']._modules['0'])
        return model 

    def _compute_accuracy(self, loader):
        num_correct = 0
        num_images = 0

        for i, (images, targets, image_ids) in enumerate(tqdm(loader)):
            images = images.cuda()
            targets = targets.cuda()
            output_dict = self.model(images)

            if self.args.architecture_type =='vitol':
                pred = output_dict.argmax(dim=1)
            else:
                pred = output_dict['logits'].argmax(dim=1)

            num_correct += (pred == targets).sum().item()
            num_images += images.size(0)

        classification_acc = num_correct / float(num_images) * 100
        return classification_acc

    def evaluate(self, split):
        print("Evaluating model on dataset {}".format(self.args.dataset_name))
        self.model.eval()

        accuracy = self._compute_accuracy(loader=self.loaders[split])
        print("Accuracy", accuracy)
        self.performance_meters[split]['classification'].update(accuracy)

        cam_computer = CAMComputer(
            model=self.model,
            loader=self.loaders[split],
            metadata_root=os.path.join(self.args.metadata_root, split),
            mask_root=self.args.mask_root,
            iou_threshold_list=self.args.iou_threshold_list,
            dataset_name=self.args.dataset_name,
            split=split,
            args = self.args,
            cam_curve_interval=self.args.cam_curve_interval,
            multi_contour_eval=self.args.multi_contour_eval,
            log_folder=self.args.log_folder,
        )
        cam_performance, top1_loc_acc = cam_computer.compute_and_evaluate_cams()

        if self.args.multi_iou_eval:
            loc_score = np.average(cam_performance)
        else:
            loc_score = cam_performance[self.args.iou_threshold_list.index(50)]

        print(loc_score, top1_loc_acc)

        self.performance_meters[split]['localization'].update(loc_score)
        self.performance_meters[split]['top1_loc_acc'].update(top1_loc_acc)

        if self.args.dataset_name in ('CUB', 'ILSVRC'):
            for idx, IOU_THRESHOLD in enumerate(self.args.iou_threshold_list):
                self.performance_meters[split][
                    'localization_IOU_{}'.format(IOU_THRESHOLD)].update(
                    cam_performance[idx])

    def print_performances(self, split):
        for metric in self._EVAL_METRICS:
            current_performance = \
                self.performance_meters[split][metric].current_value
            if current_performance is not None:
                print("Split {}, metric {}, current value: {}".format(
                    split, metric, current_performance))
                if split != 'test':
                    print("Split {}, metric {}, best value: {}".format(
                        split, metric,
                        self.performance_meters[split][metric].best_value))
                    print("Split {}, metric {}, best epoch: {}".format(
                        split, metric,
                        self.performance_meters[split][metric].best_epoch))

    def load_checkpoint_eval(self, checkpoint_type):
        
        checkpoint_path = os.path.join(
            self.args.base_model_dir, self.args.ckpt_name)

        if os.path.isfile(checkpoint_path):
            checkpoint = torch.load(checkpoint_path)

            self.model.load_state_dict(checkpoint['state_dict'], strict=True)
            # self.model.load_state_dict(checkpoint, strict=True)
            # self.model.load_state_dict(checkpoint['model'], strict=True)
            print("Check {} loaded.".format(checkpoint_path))
        else:
            raise IOError("No checkpoint {}.".format(checkpoint_path))

infer = ViTOLInference()
infer.load_checkpoint_eval(checkpoint_type=infer.args.eval_checkpoint_type)
infer.evaluate(split='test')
infer.print_performances('test')


