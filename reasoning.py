#!/usr/bin/env python
# coding: utf-8

# ## Load necessary modules
# import keras
import keras

import sys

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color
# from keras_retinanet.utils.gpu import setup_gpu

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import csv
import os
import numpy as np
import time
import argparse

# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf

# use this to change which GPU to use
# gpu = 0

# set the modified tf session as backend in keras
# setup_gpu(gpu)

# 读取一些参数
parser = argparse.ArgumentParser()

# 图片文件路径
parser.add_argument('--pictureDir', '-p', type=str, required=True)
# 输出路径（不以'/'结尾）
parser.add_argument('--outputDir', '-o', type=str, required=True)
# 模型文件路径
parser.add_argument('--modelDir', '-m', type=str, required=True)

args = parser.parse_args()


# ## Load RetinaNet model


# adjust this to point to your downloaded/trained model
# models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases
# 直接指定模型的绝对路径
model_path = args.modelDir

# load retinanet model
model = models.load_model(model_path, backbone_name='resnet50')

# if the model is not converted to an inference model, use the line below
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
model = models.convert_model(model)

# print(model.summary())

# load label to names mapping for visualization purposes
labels_to_names = {0: 'small-vehicle', 1: 'plane', 2: 'harbor'}


# ## Run detection on example


# load image
image = read_image_bgr(args.pictureDir)

# copy to draw on
draw = image.copy()
draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

# preprocess image for network
image = preprocess_image(image)
image, scale = resize_image(image)

# process image
start = time.time()
boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
print("processing time: ", time.time() - start)


# correct for image scale
boxes /= scale
target = []
# visualize detections
for box, score, label in zip(boxes[0], scores[0], labels[0]):
    # scores are sorted so we can break
    if score < 0.5:
        break
    
    target.append([args.pictureDir, box[0], box[1], box[2], box[3], score, labels_to_names[label] ])

    color = label_color(label)
    
    b = box.astype(int)
    draw_box(draw, b, color=color)
    
    caption = "{} {:.3f}".format(labels_to_names[label], score)
    draw_caption(draw, b, caption)

# 结果写入csv
f = open(args.outputDir + '/' + 'target.csv','w')
f_csv = csv.writer(f)
f_csv.writerows(target)
f.close()

plt.figure(figsize=(10, 10))
plt.axis('off')
plt.imshow(draw)
plt.show()