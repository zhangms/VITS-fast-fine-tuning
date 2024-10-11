import base64
import os
import time
import uuid
from datetime import datetime

from fastapi import HTTPException, FastAPI
from pydantic import BaseModel

from predictor import Predictor


class AudGenerateRequest(BaseModel):
    text: str
    speaker_id: str
    trace_id: str


class AudGenerateResponseBase64(BaseModel):
    audio: str


app = FastAPI()
predictor = Predictor("./OUTPUT_MODEL/G_latest.pth", "./OUTPUT_MODEL/config.json")


def encode_audio(audio_file):
    with open(audio_file, "rb") as f:
        encoded_content = base64.b64encode(f.read())
    return encoded_content


@app.get("/api/check-health")
def check_health():
    return "OK"


@app.post("/audgeneratebase64", response_model=AudGenerateResponseBase64)
async def audio_generate_base64(req: AudGenerateRequest):
    try:
        print("TTS_BEGIN:[TraceId:{}] req:{}, time:{}".format(req.trace_id, req, datetime.now()))
        st = time.time()
        text = req.text
        speaker_id = req.speaker_id
        output_path = "./OUTPUT/" + str(uuid.uuid4()) + ".mp3"
        predictor.tts_fn(text, speaker_id=speaker_id, output_path=output_path)
        audio_content = encode_audio(output_path)
        os.remove(output_path)
        print("TTS_END:[TraceId:{}] req:{}, rt:{} s.".format(req.trace_id, req, (time.time() - st)))
        return {"audio": audio_content, }
    except Exception as e:
        print("TTS_ERR:[TraceId:{}] req:{}, time:{}".format(req.trace_id, req, datetime.now()))
        raise HTTPException(status_code=500, detail=str(e))
