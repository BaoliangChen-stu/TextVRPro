import torch
import clip
from PIL import Image
import json
import shutil
import os
import cv2
import numpy as np
import logging

def logger_config(log_path, logging_name):
    '''
    配置log
    :param log_path: 输出log路径
    :param logging_name: 记录中name，可随意
    :return:
    '''
    '''
    logger是日志对象，handler是流处理器，console是控制台输出（没有console也可以，将不会在控制台输出，会在日志文件中输出）
    '''
    # 获取logger对象,取名
    logger = logging.getLogger(logging_name)
    # 输出DEBUG及以上级别的信息，针对所有输出的第一层过滤
    logger.setLevel(level=logging.DEBUG)
    # 获取文件日志句柄并设置日志级别，第二层过滤
    handler = logging.FileHandler(log_path, encoding='UTF-8')
    handler.setLevel(logging.INFO)
    # 生成并设置文件日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    # console相当于控制台输出，handler文件输出。获取流句柄并设置日志级别，第二层过滤
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    # 为logger对象添加句柄
    logger.addHandler(handler)
    logger.addHandler(console)
    return logger


def getVideo_ocvPath(json_file):
    f = open(json_file, encoding="utf-8")
    file = json.load(f)
    video_path = []
    video_name = []
    text = []
    for item in file:
        video_path.append(item['path'])
        video_name.append(item['path'].split('/')[-1].split('.')[0])
        text.append(item['captions_info'][0]['caption'])
    return video_path, video_name, text




def getSimScore(test_json):
    logger = logger_config(log_path='./result/cliplog.txt', logging_name='CLIP训练日志')
    picture_root = "/data/Video/Picture"
    video_path, videonames, textinfo = getVideo_ocvPath(test_json)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess = clip.load("ViT-L/14@336px", device=device)
    text = clip.tokenize(
        textinfo).to(device)
    simlar = np.zeros([len(video_path), len(text)])
    with torch.no_grad():
        for video_index, video in enumerate(video_path):
            video_dir = os.path.join(picture_root, video.split('.')[0].replace('/', '_'))
            images = os.listdir(video_dir)
            score = torch.zeros([len(images), len(text)])
            for idx, img in enumerate(images):
                img_path = os.path.join(video_dir, img)
                image = preprocess(Image.open(img_path)).unsqueeze(0).to(device)
                logits_per_image, logits_per_text = model(image, text)
                probs = logits_per_image.softmax(dim=-1).cpu().numpy()
                score[idx] = torch.tensor(probs[0])
            height_score, index = torch.max(score, dim=0)
            simlar[video_index] = np.array(height_score)

            information = "{}/{}".format(video_index, len(video_path))
            logger.info(information)
    np.save('./result/new_sim_matrix.npy', simlar)
    print(simlar)



# conver result from video_text to text_video
def convertRes(res_path):
    result = np.load(res_path)
    test_video = result.T
    np.save('./tesxt_video_sim_matrix.npy', test_video)
    print(test_video)

if __name__ == '__main__':
    getSimScore('./TextVR_test_rand.json')

