import json
import traceback

import uvicorn as uvicorn
from fastapi import FastAPI, Request, HTTPException


from loguru import logger
from pipelines.ad import PipelineKeeper
from utils.decode import b64_pils



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
            logger.info("Done process of traceID:{}".format(data["traceID"]))
        except Exception as e:
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=str(e))
        finally:
            pipelineKeeper.unload_lora()

        return re