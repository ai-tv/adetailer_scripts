""" utils for encoding image and arrs for saving and communicating """

import base64

import cv2
import numpy as np
from PIL import Image
import io


# @param image:ndarray
# @return b64
def ndarray_b64(image):
    image = Image.fromarray(image)
    return pil_b64(image)


# @param image:PIL image
# @return b64
def pil_b64(image):
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def pils_b64(images):
    re = []
    for img in images:
        b64 = pil_b64(img)
        re.append(b64)
    return re


# @param base64_image_data:b64 encoding
# @retun ndarray
def b64_img(base64_image_data):
    image_data = base64.b64decode(base64_image_data)
    image = Image.open(io.BytesIO(image_data))
    image = np.array(image)
    if image.shape[2] == 4:
        return image[..., :3]
    else:
        return image


def b64_pil(base64_image_data):
    image = b64_img(base64_image_data)
    image = Image.fromarray(image)
    return image


def b64_pils(b64):
    images = []
    for item in b64:
        image = b64_pil(item)
        images.append(image)
    return images
