# @Function：从之望生成的中间结果
import glob
import json
import os
from PIL import Image



path = "/mnt/lg102/zwshi/projects/playground/lc/log/"
log_list = glob.glob(path+"*.json")

img_store = {}

for item in log_list:
    with open(item) as json_file:
        data = json.load(json_file)
        c1,c2 = data["prompt"][1], data["prompt"][2]

        # out_dir = "/mnt/lg106/hou/datasets/intermediate result/"+"{} and {}".format(c1,c2)
        # if not os.path.exists(out_dir):
        #     os.mkdir(out_dir)

        tag = item.split("_")[0]
        img_path = tag+"_result.png"
        key = c1+" and "+c2
        if img_store.get(key) != None:
            img_store[key].append(img_path)
        else:
            img_store[key] = [img_path]
        
out_dir = "/mnt/lg106/hou/datasets/intermediate result/"       
for key in img_store.keys():
    if not os.path.exists(out_dir+key):
        os.mkdir(out_dir+key)
    img_list = img_store[key]
    for index,img in enumerate(img_list):
        if not os.path.exists(img):
            continue
        img = Image.open(img)
        # 随后将文件夹中所有的图片进行改名并顺序排列
        img.save(out_dir+key+"/{}.png".format(index))
# print(img_store)       


        
        
        
            
