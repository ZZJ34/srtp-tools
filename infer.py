import cv2
import sys
import os
import argparse
import numpy as np
import csv
import math
import random

# 目标检测算法输入窗口大小
MODEL_SIZE = 224

# 本脚本中 img 一律指 Image 类的实例， cvimg 才指代 opencv 图片实例。


class Image:
    def __init__(self,
                 cvimg,                     # OpenCV 图像
                 filename=None,
                 type="original",           # "original": 原图, "small": 1.0 缩放,
                                            # "middle": 0.7 缩放, "large": 0.4 缩放
                 posX=0, posY=0,            # 相对原图位移
                 original_image=None):      # 对原图 Image 实例的引用

        self.id = Image.id_counter
        Image.id_counter += 1

        self.cvimg = cvimg
        self.width = cvimg.shape[1]
        self.height = cvimg.shape[0]

        self.filename = filename
        self.type = type
        self.posX = posX
        self.posY = posY
        self.original_image = original_image


Image.id_counter = 0


def preinfer(img):
    '''在图片载入网络推理之前做些处理'''

    # 1. 切割图像，得到三个尺度的图片
    [small_images, middle_images, large_images] = split_image(
        img, output_split_images='./split')

    # 2. ...


def split_image(img, output_split_images=False):
    filename_without_extension = None \
        if img.filename == None \
        else img.filename.split('.')[0]

    small_size = int(MODEL_SIZE)
    middle_size = int(MODEL_SIZE / 0.7)
    large_size = int(MODEL_SIZE / 0.4)

    sliding_step_ratio = 0.5  # 滑动窗口步长 = 0.5 倍窗口大小

    size_names = ['small', 'middle', 'large']
    sizes = [small_size, middle_size, large_size]

    result = [[], [], []]

    for i in range(3):
        size = sizes[i]
        size_name = size_names[i]
        sliding_step = int(size * sliding_step_ratio)

        x_ith = 0
        y_ith = 0

        posX = 0
        posY = 0

        reach_x_end = False
        reach_y_end = False
        while True:
            if posY + size >= img.height:
                reach_y_end = True
                posY = img.height - size
                if posY < 0:
                    posY = 0

            x_ith = 0
            posX = 0
            reach_x_end = False

            while True:
                if posX + size >= img.width:
                    reach_x_end = True
                    posX = img.width - size
                    if posX < 0:
                        posX = 0

                posY2 = img.height if reach_y_end else posY + size
                posX2 = img.width if reach_x_end else posX + size

                img_slice_filename = None
                if filename_without_extension != None:
                    img_slice_filename = filename_without_extension + '-' + \
                        size_name + '-y' + str(y_ith) + \
                        '-x' + str(x_ith) + '.png'
                img_slice = Image(
                    cvimg=img.cvimg[posY:posY2, posX:posX2],
                    filename=img_slice_filename,
                    type=size_name,
                    posX=posX,
                    posY=posY,
                    original_image=img
                )

                result[i].append(img_slice)

                if output_split_images != None:
                    cv2.imwrite(os.path.join(output_split_images,
                                             img_slice_filename), img_slice.cvimg)

                posX += sliding_step
                x_ith += 1
                if reach_x_end:
                    break

            posY += sliding_step
            y_ith += 1
            if reach_y_end:
                break

    return result


test_img = Image(cvimg=cv2.imread(
    './tmp/images/P0000.png'), filename='P0000.png')

preinfer(test_img)