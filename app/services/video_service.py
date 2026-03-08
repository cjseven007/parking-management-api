import os
import tempfile
import shutil
import cv2
from fastapi import UploadFile


def save_upload_file_temp(file: UploadFile) -> str:
    ext = os.path.splitext(file.filename or "")[1] or ".bin"

    temp_dir = tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir=temp_dir) as tmp:
        file.file.seek(0)
        shutil.copyfileobj(file.file, tmp)
        return tmp.name


def extract_frame(video_path: str, frame_index: int):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError("Could not open uploaded video")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if frame_index < 0 or frame_index >= total_frames:
        cap.release()
        raise ValueError(f"frameIndex out of range. Total frames: {total_frames}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise ValueError("Failed to read requested frame")

    return frame, total_frames