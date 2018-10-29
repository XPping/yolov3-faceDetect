# coding: utf-8

import torch
import os
import random
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np
import cv2
import codecs
import math

class MyDataset(Dataset):
    def __init__(self, image_path, image_rects_path, transform):
        """
        :param image_path: the directory path of image. Eg. image_path="train", directory like
            [train/00001.jpg, train/00002.jpg, ...];
        :param image_rects_path: the directory path of the obeject rectange, Eg. image_path="train_rects",
            the directory like  [train_rects/00001.txt, train_rects/00002.txt, ...]
            each line in xxxxx.txt is [tag top-left-x, top-left-y, bottom-right-x, bottom-right-y\ntag top-left-x, 
            top-left-y, bottom-right-x, bottom-right-y...];tag from 0, split by \n
        :param transform: torchvision.transfomrs.
        """
        self.image_path = image_path
        self.image_rects_path = image_rects_path
        self.transform = transform

        self.max_objects_per_image = 50
        print('Start preprocessing dataset..!')
        self.preprocess()
        print('Finished preprocessing dataset..!')

    def preprocess(self):
        self.filenames = []
        self.rects = []
        dir1 = os.listdir(self.image_path)
        for d1 in dir1:
            rect_file = d1.replace('.png', '.txt').replace('.jpg', '.txt')
            if not os.path.isfile(os.path.join(self.image_rects_path, rect_file)):
                print("Can't find any rects in image: {}".format(d1))
                # self.rects[-1].append([-1, 0, 0, 0, 0])
                continue
            self.rects.append([])
            with open(os.path.join(self.image_rects_path, rect_file)) as fp:
                lines = fp.readlines()
                for line in lines:
                    line_s = np.array([float(s) for s in line.strip().split()[0:5]])
                    line_s[0] = int(line_s[0])
                    # temp_tag, top, left, bottom, right = [float(s) for s in line_s[::5]]
                    # print(line, line_s)
                    self.rects[-1].append(line_s)
            # print(self.rects[-1])
            self.filenames.append(d1)
            # for i in range(len(self.filenames)):
            #     self.getitem__(i)
        print("Image numbers: ", len(self.filenames))

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

        # pil_image = Image.fromarray(cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB))
        image = self.transform(np.uint8(padded_image))

        rects = self.rects[item]
        rects = np.array(rects, dtype=np.float32)
        tag = rects[:, 0]
        x1 = rects[:, 1]
        y1 = rects[:, 2]
        x2 = rects[:, 3]
        y2 = rects[:, 4]
        x1 += pad[1][0]
        y1 += pad[0][0]
        x2 += pad[1][0]
        y2 += pad[0][0]
        # Calculate ratios
        rects[:, 1] = (x1 + (x2 - x1)/2) / padded_w
        rects[:, 2] = (y1 + (y2 - y1)/2) / padded_h
        rects[:, 3] = (x2 - x1) / padded_w
        rects[:, 4] = (y2 - y1) / padded_h

        for i in range(len(rects)):
            if rects[i][1] > 1 or rects[i][2] > 1:
                raise "error"

        labels = np.zeros((self.max_objects_per_image, 5))
        # print(rects)
        # print(rects.shape, labels.shape, len(rects))
        labels[0:len(rects), :] = rects[0:min(self.max_objects_per_image, len(rects)), :]
        # labels[range(len(rects))[:self.max_objects_per_image]] = rects[:self.max_objects_per_image]
        return image, torch.FloatTensor(labels)

    def __len__(self):
        return len(self.filenames)
def get_loader(image_path, image_rects_path, image_size=416, batch_size=16, mode='train'):
    """Build and return data loader."""
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor()])

    dataset = MyDataset(image_path, image_rects_path, transform)

    shuffle = False
    if mode == 'train':
        shuffle = True

    data_loader = DataLoader(dataset=dataset,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return data_loader
