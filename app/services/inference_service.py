import os
import cv2
import torch
import numpy as np

MODEL_PATH = os.getenv("PARKING_MODEL_PATH", "model/parking_classifier_mobilenetv3_ts.pt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASSES = ["empty", "occupied"]

_model = None


def get_model():
    global _model

    if _model is None:
        _model = torch.jit.load(MODEL_PATH, map_location=DEVICE)
        _model.eval()

    return _model


def crop_slot(frame, slot: dict):
    points = slot.get("points")

    if points:
        pts = np.array([[int(p["x"]), int(p["y"])] for p in points], dtype=np.int32)

        x, y, w, h = cv2.boundingRect(pts)

        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [pts], 255)

        masked = cv2.bitwise_and(frame, frame, mask=mask)
        crop = masked[y:y + h, x:x + w]

        if crop.size == 0:
            raise ValueError(f"Empty polygon crop generated for slot {slot['id']}")

        return crop

    x = int(slot["x"])
    y = int(slot["y"])
    w = int(slot["w"])
    h = int(slot["h"])

    crop = frame[y:y + h, x:x + w]
    if crop.size == 0:
        raise ValueError(f"Empty bbox crop generated for slot {slot['id']}")

    return crop


def infer_slot_occupancy(crop) -> tuple[bool, float]:
    """
    Temporary heuristic baseline:
    - convert to grayscale
    - compute edge density
    - higher edge density often indicates occupied
    Replace this later with your CNN classifier.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.count_nonzero(edges)) / float(edges.size)

    occupied = edge_density > 0.08
    confidence = min(abs(edge_density - 0.08) / 0.08, 1.0)

    return occupied, round(confidence, 4)


def run_inference_for_slots(frame, slots: list[dict]) -> list[dict]:
    results = []

    for slot in slots:
        crop = crop_slot(frame, slot)

        occupied, confidence = infer_slot_occupancy(crop)

        results.append(
            {
                "slotId": slot["id"],
                "label": slot["label"],
                "occupied": occupied,
                "isAvailable": not occupied,
                "confidence": confidence,
            }
        )

    return results

# def preprocess_crop_to_array(crop: np.ndarray) -> np.ndarray:
#     gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
#     gray = cv2.resize(gray, (48, 48))
#     gray = gray.astype(np.float32) / 255.0
#     return gray

def preprocess_crop_to_array_mobilenet(crop: np.ndarray) -> np.ndarray:
    # BGR -> RGB
    rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    rgb = cv2.resize(rgb, (224, 224))
    rgb = rgb.astype(np.float32) / 255.0

    # ImageNet normalization for pretrained torchvision models
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    rgb = (rgb - mean) / std

    # HWC -> CHW
    rgb = np.transpose(rgb, (2, 0, 1))  # [3, 224, 224]
    return rgb

# @torch.no_grad()
# def infer_slots_batch(crops: list[np.ndarray]) -> list[tuple[bool, float]]:
#     model = get_model()

#     batch = [preprocess_crop_to_array(crop) for crop in crops]
#     batch = np.array(batch, dtype=np.float32)   # [N, 48, 48]
#     batch = np.expand_dims(batch, axis=1)       # [N, 1, 48, 48]

#     x = torch.from_numpy(batch).to(DEVICE)

#     logits = model(x)
#     probs = torch.softmax(logits, dim=1)
#     pred_indices = torch.argmax(probs, dim=1)

#     outputs = []
#     for i in range(len(crops)):
#         pred_idx = int(pred_indices[i].item())
#         pred_label = CLASSES[pred_idx]
#         confidence = float(probs[i, pred_idx].item())
#         occupied = pred_label == "occupied"
#         outputs.append((occupied, round(confidence, 4)))

#     return outputs

@torch.no_grad()
def infer_slots_batch_mobilenet(crops: list[np.ndarray]) -> list[tuple[bool, float]]:
    model = get_model()

    batch = [preprocess_crop_to_array_mobilenet(crop) for crop in crops]
    batch = np.array(batch, dtype=np.float32)  # [N, 3, 224, 224]
    x = torch.from_numpy(batch).to(DEVICE)

    logits = model(x)
    probs = torch.softmax(logits, dim=1)
    pred_indices = torch.argmax(probs, dim=1)

    outputs = []
    for i in range(len(crops)):
        pred_idx = int(pred_indices[i].item())
        pred_label = CLASSES[pred_idx]
        confidence = float(probs[i, pred_idx].item())
        occupied = pred_label == "occupied"
        outputs.append((occupied, round(confidence, 4)))

    return outputs

# def run_inference_for_slots_torch(frame, slots: list[dict]) -> list[dict]:
#     valid_slots = []
#     crops = []

#     for slot in slots:
#         crop = crop_slot(frame, slot)
#         valid_slots.append(slot)
#         crops.append(crop)

#     predictions = infer_slots_batch(crops)

#     results = []
#     for slot, (occupied, confidence) in zip(valid_slots, predictions):
#         results.append(
#             {
#                 "slotId": slot["id"],
#                 "label": slot["label"],
#                 "occupied": occupied,
#                 "isAvailable": not occupied,
#                 "confidence": confidence,
#             }
#         )

#     return results

def run_inference_for_slots_mobilenet(frame, slots: list[dict]) -> list[dict]:
    valid_slots = []
    crops = []

    for slot in slots:
        crop = crop_slot(frame, slot)
        valid_slots.append(slot)
        crops.append(crop)

    predictions = infer_slots_batch_mobilenet(crops)

    results = []
    for slot, (occupied, confidence) in zip(valid_slots, predictions):
        results.append(
            {
                "slotId": slot["id"],
                "label": slot["label"],
                "occupied": occupied,
                "isAvailable": not occupied,
                "confidence": confidence,
            }
        )

    return results