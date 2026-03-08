from firebase_admin import firestore
from google.cloud.firestore_v1 import GeoPoint
from app.core.firebase import db
from app.schemas.parking_area import CreateParkingAreaRequest


PARKING_AREAS = "parking_areas"


def get_all_parking_areas():
    docs = db.collection(PARKING_AREAS).stream()
    results = []

    for doc in docs:
        data = doc.to_dict() or {}
        geo = data.get("geo", {})
        gp = geo.get("geopoint")

        results.append(
            {
                "id": doc.id,
                "name": data.get("name", doc.id),
                "latitude": gp.latitude if gp else 0.0,
                "longitude": gp.longitude if gp else 0.0,
                "geohash": geo.get("geohash"),
                "capacity": data.get("capacity", 0),
                "availableCount": data.get("availableCount", 0),
                "imageWidth": data.get("layout", {}).get("imageWidth", 1920),
                "imageHeight": data.get("layout", {}).get("imageHeight", 1080),
            }
        )

    return results


def create_parking_area(payload: CreateParkingAreaRequest):
    ref = db.collection(PARKING_AREAS).document()

    doc = {
        "name": payload.name,
        "geo": {
            "geopoint": GeoPoint(payload.latitude, payload.longitude),
            "geohash": payload.geohash,
        },
        "capacity": 0,
        "availableCount": 0,
        "layout": {
            "imageWidth": payload.imageWidth,
            "imageHeight": payload.imageHeight,
        },
        "createdAt": firestore.SERVER_TIMESTAMP,
        "updatedAt": firestore.SERVER_TIMESTAMP,
    }

    ref.set(doc)

    return {
        "id": ref.id,
        "name": payload.name,
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "geohash": payload.geohash,
        "capacity": 0,
        "availableCount": 0,
        "imageWidth": payload.imageWidth,
        "imageHeight": payload.imageHeight,
    }


def get_parking_area_or_raise(area_id: str):
    ref = db.collection(PARKING_AREAS).document(area_id)
    snap = ref.get()

    if not snap.exists:
        raise ValueError("Parking area not found")

    return ref, snap.to_dict() or {}