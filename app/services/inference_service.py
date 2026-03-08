import cv2
import numpy as np


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