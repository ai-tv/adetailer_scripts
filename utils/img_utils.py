from typing import List

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageOps, ImageFilter
from insightface.app import FaceAnalysis
from utils.decode import b64_img, pil_b64, b64_pil

app = FaceAnalysis()
app.prepare(0)
dilate_erode_v = 4
mask_blur = 7


# @params:
# image：PIL.Image
# detect_num: int   检测mask的目标值
# ==================
# @return：
# masks: List[PIL.Image]
# bbox:  List[x1,y1,x2,y2]
# ？ 有的情况下，脸部会无法被检测出来，需要依赖引导图的bbox信息，后续优化 TODO
# 筛选逻辑：1.脸部conf高于阈值 2.脸部的占比为最高的前2 3.最后输出从左到右
def mask_gen(image: Image.Image, detect_num = 2):
    image = np.asarray(image)
    face_info = app.get(image[..., ::-1])
    face_info = filter(lambda x: x.det_score > 0.5, face_info)
    face_info.sort(key=lambda x: (x.bbox[2]-x.bbox[0])*(x.bbox[3]-x.bbox[1]),reversed=True)[:detect_num]
    face_info.sort(key=lambda x: x.bbox[0])
    

    masks = []
    bboxs = []

    # 先判断检测出来的face_info数量是否达到detect_num
    if len(face_info)<detect_num:
        return [None,None],[None,None]
    else:
        for face in face_info:
            mask = Image.new("L", (image.shape[1], image.shape[0]), 0)
            mask_draw = ImageDraw.Draw(mask)
            mask_draw.rectangle(face.bbox, fill=255)
            mask = np.asarray(mask)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_erode_v, dilate_erode_v))
            mask = cv2.dilate(mask, kernel)
            mask = cv2.GaussianBlur(mask, (mask_blur, mask_blur), 0)
            mask = Image.fromarray(mask)
            masks.append(mask)
            bboxs.append(face.bbox)
    
        return masks, [list(map(int, item)) for item in bboxs]

def embedding_gen(image: Image.Image):
    face = np.asarray(image)
    face_info = app.get(face[..., ::-1])
    face_embedding = face_info[0].embedding
    return face_embedding.reshape(1,-1)


def bbox_padding(
        bbox: List[int], image_size: tuple[int, int], value: int = 32
) -> List[int]:
    if value <= 0:
        return bbox

    arr = np.array(bbox).reshape(2, 2)
    arr[0] -= value
    arr[1] += value
    arr = np.clip(arr, (0, 0), image_size)
    return arr.flatten()


def composite(
        init: Image.Image,
        mask: Image.Image,
        gen: Image.Image,
        bbox_padded: tuple[int, int, int, int],
) -> Image.Image:
    img_masked = Image.new("RGBa", init.size)
    img_masked.paste(
        init.convert("RGBA").convert("RGBa"),
        mask=ImageOps.invert(mask),
    )
    img_masked = img_masked.convert("RGBA")

    size = (
        bbox_padded[2] - bbox_padded[0],
        bbox_padded[3] - bbox_padded[1],
    )
    resized = gen.resize(size)

    output = Image.new("RGBA", init.size)
    output.paste(resized, bbox_padded)
    output.alpha_composite(img_masked)
    return output.convert("RGB")

# _, bboxs = mask_gen(Image.open("../data/face2.jpg"))
# bbox_padding(bboxs[0],(1024,768))


def img_concat_compare_show(imgL1:List[Image.Image],imgL2:List[Image.Image], maxlen = 10):
    len_imgL = min(len(imgL1),10)
    single_width = imgL1[0].size[0]
    single_height = imgL1[0].size[1]
    total_width = len_imgL*single_width
    total_height = 2*single_height
    result_img = Image.new('RGB',(total_width,total_height))
    for i in range(len_imgL):
        result_img.paste(imgL1[i],(i*single_width,0))
        result_img.paste(imgL2[i],(i*single_width,single_height))
    result_img.save("compare.jpg")

