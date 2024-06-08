from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import os
import shutil
from tempfile import TemporaryDirectory
import numpy as np
import torch
from inferences.model import Inference
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()
model_path = "lite_medsam.pth"
device = "cuda" if torch.cuda.is_available() else "cpu"
inference = Inference(medsam_lite_checkpoint_path=model_path, device=device)

# Temporary directory to store uploaded files
upload_dir = TemporaryDirectory()

@app.post("/upload/")
async def upload_file(file: UploadFile = File(...)):
    file_location = os.path.join(upload_dir.name, file.filename)

    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    logger.info(f"Uploaded file in the temporary directory: {file_location}")
    return JSONResponse(content={"filename": file.filename})

@app.get("/segment/")
async def segment_file(filename: str = Query(...)):
    file_location = os.path.join(upload_dir.name, filename)

    if not os.path.exists(file_location):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        logger.info(f"Processing file: {file_location}")
        pred_save_dir = upload_dir.name
        save_overlay = False
        png_save_dir = upload_dir.name
        overwrite = True

        inference.process_file(file_location, pred_save_dir, save_overlay, png_save_dir, overwrite)
        pred_npz = np.load(os.path.join(pred_save_dir, filename), allow_pickle=True)
        pred_segs = pred_npz["segs"]
        logger.info(f"Segmentation completed for file: {file_location}")
        return JSONResponse(content={"prediction": pred_segs.tolist()})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
