import sys
sys.path.append("/home/hou/project/adetailer_scripts")

import json
import time

from utils.decode import pil_b64, b64_pils
from PIL import Image

from scripts.adetailer import proceess_adetailer
from loguru import logger
from pipelines.ad import PipelineKeeper
from utils.decode import pils_b64
import requests
import glob
import os


def test_post_app():
    
    p1 = "/mnt/lg106/hou/datasets/intermediate result/yangmi and ouyangnana"
    in_images_path = os.listdir(p1)
    in_images = [Image.open(p1+"/"+item) for item in in_images_path]
    for i in range(0,len(in_images),10):
        obj = {
        "characters": ["yangmi", "ouyangnana"],
        "images": pils_b64(in_images[i:min(len(in_images),i+10)]),
        "traceID": "1001"
        }
        start = time.time()
        # 请求接口
        response = requests.post("http://127.0.0.1:6666/get_adetailer", data=json.dumps(obj))
        if response.status_code == 200:

            result = response.json()
            images = result["images"]
            images = b64_pils(images)
            for index,img in enumerate(images):
                img.save('/mnt/lg106/hou/result/adetailer_result/yangmi and ouyangnana/'+in_images_path[index])
            cost = time.time() - start
            print("adetailer done!!! cost: {}".format(cost))
        else:
            print("failed")
    # images.append(Image.open("../data/face2.jpg"))
    # images.append(Image.open("../data/face3.jpg"))
    # images.append(Image.open("../data/face8.png"))

    # @params
    # characters:   图片上需要进行修脸的角色的姓名（从左到右）
    # images：      以b64 encoding的图片List
    # traceID：     当前任务的traceID，作为log中的tag
    

    


test_post_app()