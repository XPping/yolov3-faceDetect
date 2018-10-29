# coding: utf-8

import os
import sys
import time
import torch
import argparse
import torch
from torch.autograd import Variable
from torch import nn
import commons
from dataloader import get_loader
from models import YoloV3
import utils
import numpy as np
class Solver(object):
    def __init__(self, config):
        self.train_data_loader = get_loader(config.train_image_path, config.train_label_path,
                                            config.image_size, config.batch_size)
        self.val_data_loader = get_loader(config.val_image_path, config.val_label_path,
                                          config.image_size, config.batch_size)
        self.image_size = config.image_size
        self.lr = config.lr
        self.beta1 = config.beta1
        self.beta2 = config.beta2
        self.epoches = config.epoches
        self.yolov3_cfg = config.model_config_path
        self.num_classes = config.num_classes
        self.conf_thres = config.conf_thres
        self.nms_thres = config.nms_thres
        self.iou_thres = config.iou_thres
        self.parser_classes_tag(config.classes_file)

        self.log_path = config.log_save_path
        self.model_save_path = config.model_save_path
        commons.make_dirs([self.log_path, self.model_save_path])
        self.log_step = config.log_step
        self.model_save_step = config.model_save_step

        self.build_model()

        self.use_tensorboard = config.use_tensorboard
        self.pretrained_model = config.pretrained_model
        if config.use_tensorboard:
            self.build_tensorboard()
        if self.pretrained_model:
            print("load pretrained model in", self.pretrained_model)
            self.load_pretrain_model()

    def parser_classes_tag(self, filename):
        self.classes_tag = {}
        with open(filename, 'r') as fp:
            lines = fp.readlines()
            for i, line in enumerate(lines):
                self.classes_tag[i] = line.strip()

    def build_model(self):
        self.net = YoloV3(self.yolov3_cfg, self.num_classes)
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.net.parameters()),
                                          self.lr, [self.beta1, self.beta2])
        if torch.cuda.is_available():
            self.net.cuda()

    def build_tensorboard(self):
        from logger import Logger
        self.logger = Logger(self.log_path)

    def load_pretrain_model(self):
        # when load coco pretrained mode for face detect, we should load weight for yoloV3 except YOLOLayer weight,
        # because the classes for face detect is 1. So, only YOLOLayer weight is 255 channels, so we skip it.
        state_dict = torch.load(self.pretrained_model)
        model_state_dict = self.net.state_dict()
        requires_grad = False
        for k, v in state_dict.items():
            if len(v.size()) > 0 and v.size()[0] == 255:  # predict layer(before yolo layer) not load weight
                # print(k, v.size())
                requires_grad = True
                continue
            model_state_dict[k].copy_(state_dict[k].data)
            if not requires_grad:
                model_state_dict[k].requires_grad = False
        self.net.load_state_dict(model_state_dict)

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def val(self):
        APs = []
        for i, (images, targets) in enumerate(self.val_data_loader):
            if i == 10:
                break
            images = self.to_var(images)
            targets = self.to_var(targets)

            with torch.no_grad():
                output = self.net(images)
                output = utils.non_max_suppression(output, 80, conf_thres=self.conf_thres, nms_thres=self.nms_thres)
            # Compute average precision for each sample
            for sample_i in range(targets.size(0)):
                correct = []
                # Get labels for sample where width is not zero (dummies)
                annotations = targets[sample_i, targets[sample_i, :, 3] != 0]
                # Extract detections
                detections = output[sample_i]
                if detections is None:
                    # If there are no detections but there are annotations mask as zero AP
                    if annotations.size(0) != 0:
                        APs.append(0)
                    continue
                # Get detections sorted by decreasing confidence scores
                detections = detections[np.argsort(-detections[:, 4])]
                # If no annotations add number of detections as incorrect
                if annotations.size(0) == 0:
                    correct.extend([0 for _ in range(len(detections))])
                else:
                    # Extract target boxes as (x1, y1, x2, y2)
                    target_boxes = torch.FloatTensor(annotations[:, 1:].shape)
                    target_boxes[:, 0] = (annotations[:, 1] - annotations[:, 3] / 2)
                    target_boxes[:, 1] = (annotations[:, 2] - annotations[:, 4] / 2)
                    target_boxes[:, 2] = (annotations[:, 1] + annotations[:, 3] / 2)
                    target_boxes[:, 3] = (annotations[:, 2] + annotations[:, 4] / 2)
                    target_boxes *= self.image_size
                    detected = []
                    for *pred_bbox, conf, obj_conf, obj_pred in detections:
                        pred_bbox = torch.FloatTensor(pred_bbox).view(1, -1)
                        # Compute iou with target boxes
                        iou = utils.bbox_iou(pred_bbox, target_boxes)
                        # Extract index of largest overlap
                        best_i = np.argmax(iou)
                        # If overlap exceeds threshold and classification is correct mark as correct
                        if iou[best_i] > self.iou_thres and obj_pred == annotations[
                            best_i, 0] and best_i not in detected:
                            correct.append(1)
                            detected.append(best_i)
                        else:
                            correct.append(0)
                # Extract true and false positives
                true_positives = np.array(correct)
                false_positives = 1 - true_positives
                # Compute cumulative false positives and true positives
                false_positives = np.cumsum(false_positives)
                true_positives = np.cumsum(true_positives)
                # Compute recall and precision at all ranks
                recall = true_positives / annotations.size(0) if annotations.size(0) else true_positives
                precision = true_positives / np.maximum(true_positives + false_positives, np.finfo(np.float64).eps)
                # Compute average precision
                AP = utils.compute_ap(recall, precision)
                APs.append(AP)
        return np.mean(APs)

    def train(self):
        print("iter per epoches: ", len(self.train_data_loader))
        for e in range(self.epoches):
            for i, (images, targets) in enumerate(self.train_data_loader):
                # print(i)
                images = self.to_var(images)
                targets = self.to_var(targets)
                loss = self.net(images, targets)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                print( '[Epoch %d/%d, Batch %d/%d] [Losses: x %f, y %f, w %f, h %f, conf %f, cls %f, total %f, recall: %.5f]' %
                            (e, self.epoches, i, len(self.train_data_loader),
                             self.net.losses['x'], self.net.losses['y'], self.net.losses['w'],
                             self.net.losses['h'], self.net.losses['conf'], self.net.losses['cls'],
                             loss.item(), self.net.losses['recall']))

                self.net.seen += images.size(0)
                # Log
                Log_loss = {}
                Log_loss['train_loss'] = loss.item()
                # Print log
                if (i + 1) % self.log_step == 0:
                    APs = self.val()
                    Log_loss['APs'] = APs
                    log = "Epoch: {}/{}, Iter: {}/{}".format(e + 1, self.epoches, i + 1, len(self.train_data_loader))
                    for tag, value in Log_loss.items():
                        log += ", {}: {:.4f}".format(tag, value)
                    print(log)
                if self.use_tensorboard:
                    for tag, value in Log_loss.items():
                        self.logger.scalar_summary(tag, value, e * len(self.train_data_loader) + i + 1)
            if (e+1) % self.model_save_step == 0:
                torch.save(self.net.state_dict(), os.path.join(self.model_save_path, "{}_yoloV3.pth".format(e+1)))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam optimizer beta1')
    parser.add_argument('--beta2', type=float, default=0.99, help='Adam optimizer beta2')
    parser.add_argument('--epoches', type=int, default=100, help='number of epoches')
    parser.add_argument('--log_step', type=int, default=10, help='number of step for print log')
    parser.add_argument('--model_save_step', type=int, default=5, help='number of epoches for save model')
    parser.add_argument('--batch_size', type=int, default=16, help='image batch size')
    parser.add_argument('--model_save_path', type=str, default="Out/models", help='path to save model')
    parser.add_argument('--log_save_path', type=str, default='Out/log', help='path to save log')
    parser.add_argument('--model_config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
    parser.add_argument('--use_tensorboard', type=int, default=1, help='whether use tensorboard')
    parser.add_argument('--pretrained_model', type=str, default="weights/yolov3.pth", help='yolov3 for coco detect pretrained model')
    parser.add_argument('--gpu_id', type=int, default=0, help='used gpu id')
    parser.add_argument('--conf_thres', type=int, default=0.7, help='remove score less conf_thres in non_max_suppression')
    parser.add_argument('--nms_thres', type=int, default=0.4, help='remove iour less nms_thres in non_max_suppression')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold required to qualify as detected')

    parser.add_argument('--num_classes', type=int, default=1, help="number class for object detect, default is face detect, set=80 is coco datasets")
    parser.add_argument('--classes_file', type=str, default="config/face.names", help="classes tag")
    parser.add_argument('--train_image_path', type=str, default="/home/xpp/data/face-detect/FDDB/images", help='train image path')
    parser.add_argument('--train_label_path', type=str, default="/home/xpp/data/face-detect/FDDB/labels", help='train label path')
    parser.add_argument('--val_image_path', type=str, default="/home/xpp/data/face-detect/FDDB/val", help='val image path')
    parser.add_argument('--val_label_path', type=str, default="/home/xpp/data/face-detect/FDDB/val_labels", help='val label path')
    parser.add_argument('--image_size', type=int, default=416, help='image size')

    config = parser.parse_args()
    # Set GPU id
    commons.set_gpu(config.gpu_id)
    solver = Solver(config)
    # Training
    solver.train()