import json
import time

from utils.decode import pil_b64, b64_pils
from PIL import Image

from scripts.adetailer import proceess_adetailer
from loguru import logger
from pipelines.ad import PipelineKeeper
from utils.decode import pils_b64
import requests



def test_post_app():
    images = []
    images.append(Image.open("../data/face2.jpg"))
    images.append(Image.open("../data/face3.jpg"))
    images.append(Image.open("../data/face8.png"))
    guide_bboxs = [[["x1","y1","x2","y2"]]]

    # @params
    # characters:   图片上需要进行修脸的角色的姓名（从左到右）
    # images：      以b64 encoding的图片List
    # request_id：  当前任务的id，作为log中的tag
    # guide_bboxs   List[bbox],举个例子，一张图片有两个bbox，有三张图片,list的size就是[3,2,4]
    obj = {
        "characters": ["baijingting", "yangmi"],
        "images": pils_b64(images),
        "request_id": "1001",
        "guide_bboxs": guide_bboxs, 
    }

    start = time.time()
    # 请求接口
    response = requests.post("http://192.168.110.106:6666/get_adetailer", data=json.dumps(obj))
    if response.status_code == 200:

        result = response.json()
        images = result["images"]
        images = b64_pils(images)
        for index, img in enumerate(images):
            img.save('../data/re/{}.jpg'.format(index))
        cost = time.time() - start
        print("adetailer done!!! cost: {}".format(cost))
    else:
        print("failed")


test_post_app()