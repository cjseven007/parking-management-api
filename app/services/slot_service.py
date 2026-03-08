from firebase_admin import firestore
from app.core.firebase import db

PARKING_AREAS = "parking_areas"
SLOTS = "slots"


def replace_slots_for_area(area_id: str, slots: list[dict]):
    area_ref = db.collection(PARKING_AREAS).document(area_id)
    slots_ref = area_ref.collection(SLOTS)

    existing_docs = list(slots_ref.stream())
    batch = db.batch()

    for doc in existing_docs:
        batch.delete(doc.reference)

    for idx, slot in enumerate(slots, start=1):
        slot_ref = slots_ref.document(f"slot_{idx:03d}")

        batch.set(
            slot_ref,
            {
                "label": slot["label"],
                "isAvailable": True,
                "occupied": False,
                "confidence": None,
                "lastFrameIndex": None,
                "bbox": {
                    "x": slot["bbox"]["x"],
                    "y": slot["bbox"]["y"],
                    "w": slot["bbox"]["w"],
                    "h": slot["bbox"]["h"],
                },
                "points": slot["points"],
                "updatedAt": firestore.SERVER_TIMESTAMP,
            },
        )

    batch.update(
        area_ref,
        {
            "capacity": len(slots),
            "availableCount": len(slots),
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
    )

    batch.commit()

    return {"insertedSlots": len(slots)}

def get_slots_for_area(area_id: str):
    docs = db.collection(PARKING_AREAS).document(area_id).collection(SLOTS).stream()
    results = []

    for doc in docs:
        data = doc.to_dict() or {}
        bbox = data.get("bbox", {})
        points = data.get("points", [])

        results.append(
            {
                "id": doc.id,
                "label": data.get("label", doc.id),
                "isAvailable": data.get("isAvailable", False),
                "occupied": data.get("occupied"),
                "confidence": data.get("confidence"),
                "x": float(bbox.get("x", 0)),
                "y": float(bbox.get("y", 0)),
                "w": float(bbox.get("w", 0)),
                "h": float(bbox.get("h", 0)),
                "points": points,
            }
        )

    return results

def update_slot_inference_results(area_id: str, frame_index: int, results: list[dict]):
    area_ref = db.collection(PARKING_AREAS).document(area_id)
    batch = db.batch()

    available_count = 0

    for item in results:
        slot_ref = area_ref.collection(SLOTS).document(item["slotId"])
        if item["isAvailable"]:
            available_count += 1

        batch.update(
            slot_ref,
            {
                "occupied": item["occupied"],
                "isAvailable": item["isAvailable"],
                "confidence": item["confidence"],
                "lastFrameIndex": frame_index,
                "updatedAt": firestore.SERVER_TIMESTAMP,
            },
        )

    batch.update(
        area_ref,
        {
            "availableCount": available_count,
            "updatedAt": firestore.SERVER_TIMESTAMP,
        },
    )

    batch.commit()

    return {
        "availableCount": available_count,
        "occupiedCount": len(results) - available_count,
    }