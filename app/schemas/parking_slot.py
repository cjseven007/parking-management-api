from pydantic import BaseModel


class ParkingSlotResponse(BaseModel):
    id: str
    label: str
    isAvailable: bool
    x: float
    y: float
    w: float
    h: float
    occupied: bool | None = None
    confidence: float | None = None