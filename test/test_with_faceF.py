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
    refs = []
    refs.append(Image.open("../data/ref/ref_yangmi1.jpg"))

    # @params
    # images：      以b64 encoding的图片List
    # refs:         需要进行修脸的角色的ref image（从左到右）
    # traceID：     当前任务的traceID，作为log中的tag
    obj = {
        "images": pils_b64(images),
        "refs": pils_b64(refs),
        "traceID": "1001"
    }

    start = time.time()
    # 请求接口
    response = requests.post("http://192.168.110.106:6666/get_adetailer_withID", data=json.dumps(obj))
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