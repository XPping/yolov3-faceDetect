# coding: utf-8
import matplotlib
matplotlib.use('Agg')
import os
import sys
import time
import torch
import commons
from models import YoloV3
import utils
import random
import argparse
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
from PIL import Image
import numpy as np
import cv2
import codecs
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

class MyDataset(Dataset):
    def __init__(self, image_path, transform):
        """
        :param image_path: the directory path of image. Eg. image_path="train", directory like
            [train/00001.jpg, train/00002.jpg, ...];
        :param image_rects_path: the directory path of the obeject rectange, Eg. image_path="train_rects",
            the directory like  [train_rects/00001.txt, train_rects/00002.txt, ...]
            each line in xxxxx.txt is [tag top left bottom right\ntag top left bottom right...];tag from 0
        :param transform: torchvision.transfomrs.
        """
        self.image_path = image_path
        self.transform = transform

        print('Start preprocessing dataset..!')
        self.preprocess()
        print('Finished preprocessing dataset..!')

    def preprocess(self):
        self.filenames = []
        dir1 = os.listdir(self.image_path)
        for d1 in dir1:
            self.filenames.append(d1)
    def __getitem__(self, item):
        # Read image
        image = np.array(Image.open(os.path.join(self.image_path, self.filenames[item])))
        # Convert gray to 3 channels
        if len(image.shape) == 2:
            image = np.repeat(image, 3, axis=2)
        h, w, _ = image.shape

        sub_h_w = np.abs(h - w)
        pad1, pad2 = sub_h_w // 2, sub_h_w - sub_h_w // 2
        pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))

        padded_image = np.pad(image, pad, 'constant', constant_values=127.5)
        padded_h, padded_w, _ = padded_image.shape

        return self.transform(np.uint8(padded_image)), self.filenames[item]
    def __len__(self):
        return len(self.filenames)

def get_loader(image_path, image_size=416, batch_size=16):
    """Build and return data loader."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])

    dataset = MyDataset(image_path, transform)

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=False)
    return data_loader

class Solver(object):
    def __init__(self, config):
        self.image_size = config.image_size
        self.num_classes = config.num_classes
        self.conf_thres = config.conf_thres
        self.nms_thres = config.nms_thres
        self.image_path = config.image_path
        # Build detect image loader
        data_loader = get_loader(self.image_path, self.image_size, config.batch_size)
        self.data_loader = data_loader
        # The path to save detect result
        self.dst_path = config.dst_path
        if not os.path.isdir(self.dst_path):
            os.makedirs(self.dst_path)
        # yolov3 network
        self.yolov3_cfg = config.model_config_path
        # Parser category tag
        self.parser_category_tag(config.classes_file)
        # Pretrained model save path
        self.model_save_path = config.model_save_path
        # Build yolov3
        self.build_model()
        # Load pretrained model
        self.load_pretrain_model()
        # Bounding-box colors
        cmap = plt.get_cmap('tab20b')
        self.colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    def parser_category_tag(self, filename):
        self.classes_tag = {}
        with open(filename, 'r') as fp:
            lines = fp.readlines()
            for i, line in enumerate(lines):
                self.classes_tag[i] = line.strip()

    def build_model(self):
        self.net = YoloV3(self.yolov3_cfg, self.num_classes)
        if torch.cuda.is_available():
            self.net.cuda()

    def load_pretrain_model(self):
        # self.net.load_weights("/home/xpp/code/object-detector/yolov3/PyTorch-YOLOv3/weights/yolov3.weights")
        # torch.save(self.net.state_dict(), "/home/xpp/code/object-detector/yolov3/PyTorch-YOLOv3/weights/yolov3.pth")
        self.net.load_state_dict(torch.load(self.model_save_path))

    def to_var(self, x, volatile=False):
        if torch.cuda.is_available():
            x = x.cuda()
        return Variable(x, volatile=volatile)

    def test(self):
        print("iter per epochde: ", len(self.data_loader))
        for i, (images, images_path) in enumerate(self.data_loader):
            # print(images.shape)
            images = self.to_var(images)
            predict = self.net(images)
            # print(predict.shape)
            predict = utils.non_max_suppression(predict.cpu().data, len(self.classes_tag), self.conf_thres, self.nms_thres)
            self.save_predict_result(predict, images_path)
            # raise
    def save_predict_result(self, predict, images_path):
        for i, pre in enumerate(predict):
            # (x1, y1, x2, y2, conf, cls_conf, cls_pred)
            # Read image
            image = np.array(Image.open(os.path.join(self.image_path, images_path[i])))
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(image)
            if pre is not None:
                # Convert gray to 3 channels
                if len(image.shape) == 2:
                    image = np.repeat(image, 3, axis=2)
                h, w, _ = image.shape
                sub_h_w = np.abs(h - w)
                pad1, pad2 = sub_h_w // 2, sub_h_w - sub_h_w // 2
                pad = ((pad1, pad2), (0, 0), (0, 0)) if h <= w else ((0, 0), (pad1, pad2), (0, 0))
                padded_h = h + pad[0][0] + pad[0][1]
                padded_w = w + pad[1][0] + pad[1][1]
                # padded_image = np.pad(image, pad, 'constant', constant_values=128) / 255
                # padded_h, padded_w, _ = padded_image.shape

                bbox_colors = random.sample(self.colors, len(pre))
                for ii, (x1, y1, x2, y2, conf, cls_conf, cls_pred) in enumerate(pre.numpy()):
                    x1 = x1 / self.image_size * padded_w
                    x2 = x2 / self.image_size * padded_w
                    y1 = y1 / self.image_size * padded_h
                    y2 = y2 / self.image_size * padded_h
                    x1 -= pad[1][0]
                    x2 -= pad[1][0]
                    y1 -= pad[0][0]
                    y2 -= pad[0][0]
                    color = bbox_colors[int(np.where(pre[:, -1].unique() == int(cls_pred))[0])]
                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2,
                                             edgecolor=color,
                                             facecolor='none')
                    # Add the bbox to the plot
                    ax.add_patch(bbox)
                    # Add label
                    plt.text(x1, y1, s=self.classes_tag[int(cls_pred)], color='white', verticalalignment='top',
                             bbox={'color': color, 'pad': 0})
            plt.axis('off')
            plt.savefig(os.path.join(self.dst_path, images_path[i]))
            plt.close()
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=16, help='image batch size')
    parser.add_argument('--model_config_path', type=str, default='config/yolov3.cfg', help='path to model config file')
    parser.add_argument('--model_save_path', type=str, default="weights/face.pth", help='path to model save path')
    parser.add_argument('--conf_thres', type=int, default=0.7, help='remove score less conf_thres in non_max_suppression')
    parser.add_argument('--nms_thres', type=int, default=0.4, help='remove iour less nms_thres in non_max_suppression')
    parser.add_argument('--num_classes', type=int, default=1, help="number class for object detect, default is face detect, set=80 is coco datasets")
    parser.add_argument('--classes_file', type=str, default="config/face.names", help="classes tag")
    parser.add_argument('--image_path', type=str, default="/home/xpp/data/face-detect/FDDB/val", help='detect image path')
    parser.add_argument('--dst_path', type=str, default="/home/xpp/data/face-detect/self-detect-test", help='detect result save path')
    parser.add_argument('--image_size', type=int, default=416, help='image size')
    parser.add_argument('--gpu_id', type=int, default=0, help='used gpu id')

    config = parser.parse_args()
    # Set GPU id
    commons.set_gpu(config.gpu_id)
    solver = Solver(config)
    # Detect
    solver.test()