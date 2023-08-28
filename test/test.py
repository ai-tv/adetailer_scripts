import json
import time

from utils.decode import pil_b64, b64_pils
from PIL import Image

from scripts.adetailer import proceess_adetailer
from loguru import logger
from pipelines.ad import PipelineKeeper
from utils.decode import pils_b64
import requests



# logger.add("adetailer_server.log")
#
#
#
# image = Image.open("../data/face2.jpg")
# encode_image = pil_b64(image)
# prompt = 'ouyangnana, <lora:exp0814_ouyangnana_single-000005:0.9> [SEP] baijingting, <lora:exp0814_baijingting_single-000005:0.9>'
# proceess_adetailer({"img": encode_image, "prompt": prompt})


# prompt = 'ouyangnana, <lora:exp0814_ouyangnana_single-000005:0.9> [SEP] baijingting, <lora:exp0814_baijingting_single-000005:0.9>'
# images = []
# images.append(Image.open("../data/face2.jpg"))
# images.append(Image.open("../data/face3.jpg"))
# images.append(Image.open("../data/face8.png"))
#
# pk = PipelineKeeper()
# pk.init_pipeline(2)
# re = pk.process(images, "baijingting [SEP] yangmi", "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)")
# for index, item in enumerate(re):
#     item.save("re{}.jpg".format(index))



def test_post_app():
    images = []
    images.append(Image.open("../data/face2.jpg"))
    images.append(Image.open("../data/face3.jpg"))
    images.append(Image.open("../data/face8.png"))

    obj = {
        "characters": ["baijingting", "yangmi"],
        "images": pils_b64(images),
        "traceID": "1001"
    }

    start = time.time()
    response = requests.post("http://127.0.0.1:6666/get_adetailer", data=json.dumps(obj))
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
