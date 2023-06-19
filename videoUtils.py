import torch
import clip
from PIL import Image
import json
import shutil
import os
import cv2
import numpy as np
import logging

def copyVideo(vide_name, org_root, new_root):
    for i, name in enumerate(vide_name):
        name = name.replace('\\', '///')
        org_path = os.path.join(org_root, name)
        new_path = os.path.join(new_root, name)
        shutil.copyfile(org_path, new_path)
        print("{}/{}".format(i, len(vide_name)))

def save_image(image, addr, num):
    address = addr + "/frame" + str(num) + '.jpg'
    cv2.imwrite(address, image)

def video2pic(video_path, video_root, pic_root):
    for idx, name in enumerate(video_path):
        read_video_path = os.path.join(video_root, name)
        save_root = os.path.join(pic_root, name.split('.')[0].replace('/', '_'))
        videoCapture = cv2.VideoCapture(read_video_path)
        success, frame = videoCapture.read()
        i = 0
        timeF = 1
        j = 0
        while success:
            i = i + 1
            if (i % timeF == 0):
                j = j + 1
                save_image(frame, save_root, j)  # 视频截成图片存放的位置
                # print('save image:', i)
            success, frame = videoCapture.read()
        print("{}/{}".format(idx, len(video_path)))







