import json
import traceback

import uvicorn as uvicorn
from fastapi import FastAPI, Request, HTTPException

from loguru import logger
from pipelines.ad import PipelineKeeper
from utils.decode import b64_pils, pils_b64
import sys

sys.path.append("D:/adetailer_scripts/networks")

# step.1    先启动log
logger.add("./log/process.log")
logger.info("====adetailer service start====")

# step.2    加载配置文件
# model_config_path = "configs/config_model.json"
# adetailer_config_path = "configs/config_adetailer.json"

# model_config = json.load(open(model_config_path))
# adetailer_config = json.load(open(adetailer_config_path))

# step.3    开始初始化pipeline
pipelineKeeper = PipelineKeeper()
pipelineKeeper.init_pipeline()

# step.4    启动fastapi
app = FastAPI()


@app.post("/get_adetailer")
async def get_adetailer(request: Request):
    data = await request.json()

    with pipelineKeeper.lock:
        try:
            logger.info("Get request of traceID:{}".format(data["traceID"]))
            images_b64 = data["images"]
            images = b64_pils(images_b64)
            logger.info("Process images of traceID:{}".format(data["traceID"]))
            re = pipelineKeeper.process(images, data["characters"])
            re = pils_b64(re)
            re = {"images": re}
            logger.info("Done process of traceID:{}".format(data["traceID"]))
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            pipelineKeeper.unload_lora()

        return re


@app.post("/get_adetailer_withID")
async def get_adetailer(request: Request):
    data = await request.json()

    with pipelineKeeper.lock:
        try:
            logger.info("Get request of traceID:{}".format(data["traceID"]))
            images_b64 = data["images"]
            refs_b64 = data["refs"]
            images = b64_pils(images_b64)
            refs = b64_pils(refs_b64)
            logger.info("Process images of traceID:{}".format(data["traceID"]))
            re = pipelineKeeper.process_with_ref(images, refs)
            re = pils_b64(re)
            re = {"images": re}
            logger.info("Done process of traceID:{}".format(data["traceID"]))
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            pipelineKeeper.unload_lora()

        return re


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=6666, workers=1)
