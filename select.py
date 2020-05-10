from cv2 import cv2 
import sys
import os
import argparse
import numpy as np
import csv

# 过滤一下文件名
def picturesFilter(item):
  return not item.startswith('.')
# 计算欧式距离(像素距离)
def cal_distance(point1, point2):
  dis = np.sqrt(np.sum(np.square(point1[0]-point2[0])+np.square(point1[1]-point2[1])))
  return dis
# 基于海伦公式计算不规则四边形的面积(像素面积)
def helen_formula(coord):
    coord = np.array(coord).reshape((4,2))
    # 计算各边的欧式距离
    dis_1 = cal_distance(coord[0], coord[1])
    dis_2 = cal_distance(coord[1], coord[2])
    dis_3 = cal_distance(coord[2], coord[3])
    dis_5 = cal_distance(coord[3], coord[1])
    dis_4 = cal_distance(coord[0], coord[3])
    p1 = (dis_1+dis_4+dis_5)*0.5
    p2 = (dis_2+dis_3+dis_5)*0.5
    # 计算两个三角形的面积
    area1 = np.sqrt(p1*(p1-dis_1)*(p1-dis_4)*(p1-dis_5))
    area2 = np.sqrt(p2*(p2-dis_2)*(p2-dis_3)*(p2-dis_5))
    return area1+area2


def main():
  # csv 文件的标题
  headers= ['picDir','x1','y1','x2','y2','type']
  # 缩小有用的标注
  useful = []
  # 缩小无用的标注
  abandon = []

  # 读取启动参数
  # 参数按照顺序依次为：图片文件夹绝对路径、标注文件夹绝对路径、模型输入大小、阈值大小、输出绝对路径
  parser = argparse.ArgumentParser()

  parser.add_argument('--pictureDir', '-p', type=str, required=True)
  parser.add_argument('--labelDir', '-l', type=str, required=True)
  parser.add_argument('--modelSize', '-m', type=float, required=True)
  parser.add_argument('--threshold', '-t', type=float, required=True)
  parser.add_argument('--outputDir', '-o', type=str, required=True)
  # 从标注文件的第几行读取
  parser.add_argument('--readFromLine', '-r', type=int,required=True)
  
  args = parser.parse_args()

  pictureDir = args.pictureDir
  print('图片路径:', pictureDir)
  labelDir = args.labelDir
  print('标注路径:', labelDir)
  modelSize = args.modelSize
  print('模型大小:', modelSize)
  threshold = args.threshold
  print('阈值大小:', threshold)
  outputDir = args.outputDir
  print('输出路径:', outputDir)
  
  print('\n..............清洗开始............')
  # 获取图片文件夹下面的所有图片名称
  pictureList = list(filter(picturesFilter, os.listdir(pictureDir)))
  for pictureItem in pictureList:

    isUseFul = False
    print(pictureItem)

    # 读取图片
    pic = cv2.imread(pictureDir + '/'+ pictureItem) 
    shape = pic.shape
    X = shape[0]
    Y = shape[1]
    ratio = float(shape[1]/modelSize) if shape[0] < shape[1] else float(shape[0]/modelSize)


    # 读取标签 labels 文件
    f = open(labelDir + '/' + pictureItem.split('.')[0] + '.txt', 'r')
    labelTxt = f.readlines()[args.readFromLine -1 :]
    f.close()

    # 处理标签
    for label in labelTxt:
      x1 = int(float(label.split(' ')[0]))
      y1 = int(float(label.split(' ')[1]))

      x2 = int(float(label.split(' ')[2]))
      y2 = int(float(label.split(' ')[3]))

      x3 = int(float(label.split(' ')[4]))
      y3 = int(float(label.split(' ')[5]))

      x4 = int(float(label.split(' ')[6]))
      y4 = int(float(label.split(' ')[7]))
      
      # 目标种类
      kind = label.split(' ')[8]
      # ['small-vehicle', 'plane', 'harbor']
      if not (kind == 'small-vehicle' or kind == 'plane' or kind == 'harbor'):
        continue

      if max([x1, x2, x3, x4]) > X-1:
        xmax = X - 1
      else:
        xmax = max([x1, x2, x3, x4]) - 1

      if max([y1, y2, y3, y4]) > Y-1:
        ymax = Y - 1
      else:
        ymax = max([y1, y2, y3, y4]) -1
      
      xmin = min([x1, x2, x3, x4])
      ymin = min([y1, y2, y3, y4])
      
      # 目标的框框，感觉很牛逼的样子(标注所有)
      # cv2.rectangle(pic, (xmin,ymin), (xmax,ymax), (0,255,0), 2)

      # 计算目标的像素面积
      area = helen_formula([x1, y1, x2, y2, x3, y3, x4, y4]) /(ratio * ratio)
      
      # 根据阈值判断目标是否有用
      if area >= threshold and (xmax - xmin)/ratio >= threshold and (ymax - ymin)/ratio >= threshold:
        isUseFul = True
        useful.append([outputDir + '/images/' + pictureItem, xmin, ymin, xmax, ymax, kind])
        # 目标的框框，感觉很牛逼的样子(标注有用)
        # cv2.rectangle(pic, (xmin,ymin), (xmax,ymax), (0,255,0), 2)
      else:
        abandon.append([pictureDir + '/' + pictureItem, xmin, ymin, xmax, ymax, kind])
    
    if isUseFul:
      cv2.imwrite(outputDir + '/images/' + pictureItem, pic)
    #cv2.imwrite('ZZJ' + pictureItem, pic)
  
  print('..............清洗结束............')
  print('\n..............数据导出............')
  # 写入有用标注
  f = open(outputDir + '/label/' + 'useful.csv','w')
  f_csv = csv.writer(f)
  # f_csv.writerow(headers)
  f_csv.writerows(useful)
  f.close()
  # 写入无用标注
  f = open(outputDir + '/label/' + 'abandon.csv','w')
  f_csv = csv.writer(f)
  # f_csv.writerow(headers)
  f_csv.writerows(abandon)
  f.close()

  print('............数据导出完成..........')

if __name__ == "__main__":
  main()