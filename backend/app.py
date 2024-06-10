from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse, Response
import os
import shutil
from tempfile import TemporaryDirectory
import torch
from inferences.inference import Inference
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
    # create a unique file name, with .npz format
    file_name = f"{file.filename.split('.npz')[0]}_{os.urandom(4).hex()}.npz"
    file_location = os.path.join(upload_dir.name, file_name)
    with open(file_location, "wb+") as file_object:
        shutil.copyfileobj(file.file, file_object)
    logger.info(f"Uploaded file in the temporary directory: {file_location}")
    return JSONResponse(content={"filename": file_name})

@app.get("/segment/")
async def segment_file(filename: str = Query(...)):
    file_location = os.path.join(upload_dir.name, filename)

    if not os.path.exists(file_location):
        raise HTTPException(status_code=404, detail="File not found")

    try:
        logger.info(f"Processing file: {file_location}")
        overwrite = True
        pred_path = os.path.join(upload_dir.name, filename.replace(".npz", "_pred.npz"))
        inference.process_file(file_location, upload_dir.name, overwrite)
        logger.info(f"File saved: {pred_path}")
        if not os.path.exists(pred_path):
            raise HTTPException(status_code=500, detail="Segmentation failed")
        logger.info(f"Segmentation completed for file: {file_location}")
        with open(pred_path, "rb") as file:
            # returns the npz file
            return Response(content=file.read(), media_type="application/octet-stream")
    except Exception as e:
        logger.error(f"Error during segmentation: {e}")
        raise HTTPException(status_code=500, detail=str(e))

