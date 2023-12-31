from __future__ import annotations


from abc import ABC, abstractmethod
from typing import  Callable,List


from PIL import Image
from loguru import logger
import numpy as np
from utils.img_utils import bbox_padding, composite
import json


logger.add("../log/process.log")


def ordinal(n: int) -> str:
    d = {1: "st", 2: "nd", 3: "rd"}
    return str(n) + ("th" if 11 <= n % 100 <= 13 else d.get(n % 10, "th"))


# 单pipeline，一次将一个batch中所有对应面部全部完成修脸，减少频繁加载卸载的问题
class AdPipelineBase(ABC):


    @property
    @abstractmethod
    def inpaint_pipeline(self) -> Callable:
        raise NotImplementedError

    # @parameter
    # ==========
    # prompt
    # negative_prompt
    # images
    # masks
    # index     当前pipeline需要处理的脸的index
    # =========
    def __call__(
            self,
            prompt_embedding=None,
            negative_prompt_embedding=None,
            images: List[Image.Image] = None,
            masks: List[Image.Image] = None,
            bboxs: List[np.ndarray] = None,
            index: int = 0,
            log_path: str =None

    ):
        # init_images = []
        final_images = []
        # 每张图片进行一次修脸
        for i, init_image in enumerate(images):
            # init_images.append(init_image.copy())

            mask = masks[i][index]
            bbox = bboxs[i][index]
            # 要是检测到mask有None，就是原图被检测出来的脸部数目对不上，放弃生成，直接返回原图
            if mask == None:
                final_images.append(init_image)
                continue


            bbox_padded = bbox_padding(bbox, init_image.size)
            bbox_padded = tuple(bbox_padded)
            crop_image = init_image.crop(bbox_padded)
            crop_mask = mask.crop(bbox_padded)
            inpaint_args = self.get_inpaint_args()
            inpaint_args["image"] = crop_image
            inpaint_args["mask_image"] = crop_mask
            crop_image.save(log_path+"/crop_{}_{}.jpg".format(i,index))
            crop_mask.save(log_path+"/cropM_{}_{}.jpg".format(i,index))
            # inpaint_args["width"] = init_image.size[0]
            # inpaint_args["height"] = init_image.size[1]
            inpaint_args["prompt_embeds"] = prompt_embedding
            inpaint_args["negative_prompt_embeds"] = negative_prompt_embedding

            # 将crop_image进行inpaint
            inpaint_output = self.inpaint_pipeline(**inpaint_args)
            inpaint_image: Image.Image = inpaint_output[0][0]
            inpaint_image.save(log_path+"/inpaint_{}_{}.jpg".format(i,index))
            final_image = composite(
                init=init_image,
                mask=mask,
                gen=inpaint_image,
                bbox_padded=bbox_padded,
            )
            final_images.append(final_image)

            # =================

        return final_images

    def get_inpaint_args(self):

        config = json.load(open("./configs/config_ad.json"))
        return config["inpaint_args"]
