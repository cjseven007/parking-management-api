from pydantic import BaseModel, Field


class CreateParkingAreaRequest(BaseModel):
    name: str = Field(..., min_length=1)
    latitude: float
    longitude: float
    geohash: str | None = None
    imageWidth: int = 1920
    imageHeight: int = 1080


class ParkingAreaResponse(BaseModel):
    id: str
    name: str
    latitude: float
    longitude: float
    geohash: str | None = None
    capacity: int
    availableCount: int
    imageWidth: int
    imageHeight: int