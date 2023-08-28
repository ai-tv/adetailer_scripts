from utils.decode import pil_b64
from PIL import Image
from utils.mask import imgMaskGen
from scripts.adetailer import proceess_adetailer

image = Image.open("data/face6.jpg")
encode_image = pil_b64(image)
prompt = 'ouyangnana, <lora:exp0814_ouyangnana_single-000005:0.9> [SEP] baijingting, <lora:exp0814_baijingting_single-000005:0.9>'
proceess_adetailer({"img": encode_image, "prompt": prompt})
