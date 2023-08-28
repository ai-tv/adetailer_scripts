import json
import time

import requests

from utils.img_utils import mask_gen
from dataclasses import dataclass, asdict
from utils.decode import b64_img, b64_pil

i2iConfig = {
    "init_images": [
        "string"
    ],
    "resize_mode": 0,
    "denoising_strength": 0.45,
    "mask": "string",
    "mask_blur": 4,
    "mask_blur_x": 4,
    "mask_blur_y": 4,
    "inpainting_fill": 1,
    "inpaint_full_res": 1,
    "inpaint_full_res_padding": 32,
    "sampler_name": "DPM++ SDE Karras",
    "batch_size": 1,
    "steps": 10,
    "sampler_index": "DPM++ SDE Karras",
    "seed_resize_from_h": 0,
    "seed_resize_from_w": 0,
    "image_cfg_scale": 7,
    "negative_prompt": "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)"


}


@dataclass
class AdetailerParams:
    img: str = None
    prompt: str = None
    negative_prompt: str = None


# @param:    img:b64 需要进行修脸的图片；prompt：str adetailer需要使用的prompt，用[sep]完成分割
# !deprecated
def proceess_adetailer(params):
    params = AdetailerParams(**params)

    # step.1 现将prompt以[sep]完成切割：
    prompt = params.prompt
    subprompt_list = prompt.split('[SEP]')

    # step.2 生成mask
    masks = imgMaskGen(params.img)

    # step.3 调用webui的fastapi接口，对图片进行inpaint
    img_i2i = params.img
    img_decode = b64_img(img_i2i)
    i2iConfig['width'] = img_decode.shape[1]
    i2iConfig['height'] = img_decode.shape[0]

    start = time.time()
    for index, mask in enumerate(masks):

        i2iConfig['mask'] = mask
        i2iConfig['init_images'] = [img_i2i]
        i2iConfig['prompt'] = subprompt_list[index]


        response = requests.post("http://127.0.0.1:7861/sdapi/v1/img2img", data=json.dumps(i2iConfig))
        if response.status_code == 200:

            result = response.json()
            image = result["images"][0]
            img_i2i = image

        else:
            print("Error:", response.text)
    img = b64_pil(img_i2i)
    cost = time.time() - start
    output_name = 'result.png'
    img.save(output_name)
    print("cost %.3fs, save result to %s" % (cost, output_name,))
