from __future__ import annotations

import json
from threading import Lock
from functools import cached_property
from typing import List


import torch
from diffusers import (
    StableDiffusionInpaintPipeline,
    StableDiffusionPipeline,
    DPMSolverSDEScheduler,
    DPMSolverMultistepScheduler,
)

from PIL import Image
from pipelines.ad_base import AdPipelineBase

from utils.img_utils import mask_gen, embedding_gen
from utils.lora_loader import LoraLoader
from utils.long_prompt_weighting import get_weighted_text_embeddings



class AdPipeline(AdPipelineBase, StableDiffusionPipeline):

    @cached_property
    def inpaint_pipeline(self):
        return StableDiffusionInpaintPipeline(
            vae=self.vae,
            text_encoder=self.text_encoder,
            tokenizer=self.tokenizer,
            unet=self.unet,
            scheduler=self.scheduler,
            safety_checker=self.safety_checker,
            feature_extractor=self.feature_extractor,
            requires_safety_checker=False,
        )


class PipelineKeeper:

    def __init__(self):
        self.pipeline_repo = []
        self.lora_onload_keys = []
        self.loraloader = LoraLoader()
        self.lock = Lock()
        self.config_lora = json.load(open("./configs/config_lora.json"))
        self.config_model = json.load(open("./configs/config_model.json"))
        self.id_mlp = torch.load(self.config_model['id_mlp']).to("cuda").eval()

    def init_pipeline(self, pipeline_num=2):
        SCHEDULER_LINEAR_START = 0.00085
        SCHEDULER_LINEAR_END = 0.0120
        SCHEDULER_TIMESTEPS = 1000
        SCHEDLER_SCHEDULE = "scaled_linear"

        for i in range(pipeline_num):
            # print(self.config_model["local_base"])
            p = AdPipeline.from_pretrained("emilianJR/chilloutmix_NiPrunedFp32Fix", torch_dtype=torch.float16)
            p.scheduler = DPMSolverMultistepScheduler(
                num_train_timesteps=SCHEDULER_TIMESTEPS,
                beta_start=SCHEDULER_LINEAR_START,
                beta_end=SCHEDULER_LINEAR_END,
                beta_schedule=SCHEDLER_SCHEDULE,
                algorithm_type='dpmsolver++')
            p = p.to("cuda")
            self.add_pipeline(p)
        self.lora_onload_keys = []
        self.loraloader = LoraLoader()

    # 存储了lora_key,之后可以判断是否需要卸载和重载 TODO
    def load_lora_byconfig(self, lora_keys):

        lora_loader = self.loraloader
        for index, item in enumerate(lora_keys):
            lora_loader.load_lora_weights(self.pipeline_repo[index], self.config_lora[lora_keys[index]], 1, 'cuda',
                                          torch.float32)

        for key in lora_keys:
            self.lora_onload_keys.append(key)

    def process(self, images: List[Image.Image], characters: List[str]):
        # todo 根据prompt加载lora
        prompts = ["{}, master piece, detailed face".format(c) for c in characters]
        negative_prompt = "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)"
        # lora_keys = []

        # 先写死需要用的lora
        lora_keys = characters
        self.load_lora_byconfig(lora_keys)

        # images生成masks和bboxs
        masks = []
        bboxs = []
        for index, item in enumerate(images):
            ms, bs = mask_gen(item)
            # 有可能会出现脸部检测缺失的问题，暂时的处理方案是如果数量对不上characters的数量,就将该照片移除infer队列
            masks.append(ms)
            bboxs.append(bs)
        input_images = images.copy()

        for index, prompt in enumerate(prompts):
            p = self.get_pipelines()[index]
            text_embedding, uncond_embedding = get_weighted_text_embeddings(p, prompt, negative_prompt)
            re = p(prompt_embedding=text_embedding, negative_prompt_embedding=uncond_embedding, images=input_images,
                   masks=masks, bboxs=bboxs, index=index)
            input_images = re

        return re

    def process_with_ref(self, images: List[Image.Image], refs: List[Image.Image]):
        prompt = "a photo of young thin face, good-looking, best quality"
        negative_prompt = "paintings, sketches, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, glans, lowres,bad anatomy,bad hands, text, error, missing fingers,extra digit, fewer digits, cropped, worstquality, low quality, normal quality,jpegartifacts,signature, watermark, username,blurry,bad feet,cropped,poorly drawn hands,poorly drawn face,mutation,deformed,worst quality,low quality,normal quality,jpeg artifacts,watermark,extra fingers,fewer digits,extra limbs,extra arms,extra legs,malformed limbs,fused fingers,too many fingers,long neck,cross-eyed,mutated hands,polar lowres,bad body,bad proportions,gross proportions,text,error,missing fingers,missing arms,missing legs,extra digit,(nsfw:1.5), (sexy)"
        self.load_lora_byconfig(["id"])

        text_embedding, uncond_embedding = get_weighted_text_embeddings(self.pipeline_repo[0], prompt, negative_prompt)
        text_embedding.to("cuda")
        uncond_embedding.to("cuda")

        masks = []
        bboxs = []
        for index, item in enumerate(images):
            ms, bs = mask_gen(item)
            # 应该将检测不成功的图片记录到log中 TODO
            masks.append(ms)
            bboxs.append(bs)
        # 需要检测当前的mask个数是否符合要求，不符合要求则取消该img的换脸操作 
        input_images = images.copy()
        for index, ref in enumerate(refs):
            p = self.get_pipelines()[0]
            con_embedding, uncond_embedding = get_weighted_text_embeddings(p, prompt, negative_prompt)
            face_embedding = embedding_gen(ref)
            face_embedding = self.id_mlp(torch.from_numpy(face_embedding).to("cuda"))
            con_embedding = torch.cat([face_embedding[None, ], text_embedding, ], dim=1)
            uncond_embedding = torch.cat([face_embedding[None, ], uncond_embedding], dim=1)
            re = p(prompt_embedding=con_embedding, negative_prompt_embedding=uncond_embedding, images=input_images,
                   masks=masks, bboxs=bboxs, index=index)
            input_images = re
        return re

    def unload_lora(self):
        for index, item in enumerate(self.lora_onload_keys):
            self.loraloader.unload_lora_weight(self.pipeline_repo[index], self.config_lora[item], 1, "cuda",
                                               torch.float32)
        self.lora_onload_keys = []

    def get_pipelines(self):
        return self.pipeline_repo

    def add_pipeline(self, pipeline: AdPipeline):
        self.pipeline_repo.append(pipeline)
