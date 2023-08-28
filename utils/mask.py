import numpy as np
import cv2
from PIL import Image, ImageDraw
from insightface.app import FaceAnalysis
from utils.decode import b64_img,pil_b64,b64_pil

app = FaceAnalysis()
app.prepare(0)
dilate_erode_v = 32
# @params:image：b64编码的图片
# @return：masks；b64编码的mask组成的list
def imgMaskGen(image):

    image = b64_img(image)
    face_info = app.get(image[..., ::-1])
    face_info.sort(key=lambda x: x.bbox[0])
    face_info = filter(lambda x:x.det_score>0.75,face_info)

    masks = []
    for face in face_info:
        mask = Image.new("L", (image.shape[1], image.shape[0]), 0)
        mask_draw = ImageDraw.Draw(mask)
        mask_draw.rectangle(face.bbox, fill=255)
        mask = np.asarray(mask)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (dilate_erode_v, dilate_erode_v))
        mask = cv2.dilate(mask, kernel)
        mask = Image.fromarray(mask)
        mask = pil_b64(mask)
        masks.append(mask)
    return masks
