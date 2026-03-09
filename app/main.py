from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from app.schemas.parking_area import CreateParkingAreaRequest
from app.services.parking_area_service import (
    get_all_parking_areas,
    create_parking_area,
    get_parking_area_or_raise,
)
from app.services.slot_service import (
    replace_slots_for_area,
    get_slots_for_area,
    update_slot_inference_results,
)
from app.services.video_service import save_upload_file_temp, extract_frame
from app.services.inference_service import run_inference_for_slots_mobilenet
from app.utils.json_parser import parse_bounding_boxes_json
import os


app = FastAPI(title="Parking Admin API")
cors_origins = os.getenv("CORS_ORIGINS", "")
origins = [o.strip() for o in cors_origins.split(",") if o.strip()]
origins.extend([
    "http://localhost:5173",
    "http://127.0.0.1:5173",
])

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "Parking Admin API is running"}


@app.get("/parking-areas")
def list_parking_areas():
    return get_all_parking_areas()


@app.post("/parking-areas")
def add_parking_area(payload: CreateParkingAreaRequest):
    try:
        return create_parking_area(payload)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/parking-areas/{area_id}/slots")
def list_slots(area_id: str):
    try:
        get_parking_area_or_raise(area_id)
        return get_slots_for_area(area_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/parking-areas/{area_id}/slots/upload-json")
async def upload_slots_json(area_id: str, file: UploadFile = File(...)):
    try:
        get_parking_area_or_raise(area_id)

        if not file.filename or not file.filename.endswith(".json"):
            raise HTTPException(status_code=400, detail="Please upload a .json file")

        raw = await file.read()
        slots = parse_bounding_boxes_json(raw)
        result = replace_slots_for_area(area_id, slots)

        return {
            "message": "Bounding boxes uploaded successfully",
            "parkingAreaId": area_id,
            **result,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/parking-areas/{area_id}/infer-from-video")
async def infer_from_video(
    area_id: str,
    frameIndex: int = Form(...),
    video: UploadFile = File(...),
):
    video_path = None

    try:
        get_parking_area_or_raise(area_id)

        slots = get_slots_for_area(area_id)
        if not slots:
            raise HTTPException(status_code=400, detail="No slots found for this parking area")

        video_path = save_upload_file_temp(video)
        frame, total_frames = extract_frame(video_path, frameIndex)

        # Model is only loaded here, lazily, on first inference request
        inference_results = run_inference_for_slots_mobilenet(frame, slots)
        counts = update_slot_inference_results(area_id, frameIndex, inference_results)

        return {
            "parkingAreaId": area_id,
            "frameIndex": frameIndex,
            "videoTotalFrames": total_frames,
            "totalSlots": len(inference_results),
            "occupiedSlots": counts["occupiedCount"],
            "availableSlots": counts["availableCount"],
            "slots": inference_results,
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    finally:
        if video_path and os.path.exists(video_path):
            os.remove(video_path)