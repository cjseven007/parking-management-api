from pydantic import BaseModel


class SlotInferenceResult(BaseModel):
    slotId: str
    label: str
    occupied: bool
    isAvailable: bool
    confidence: float


class InferenceResponse(BaseModel):
    parkingAreaId: str
    frameIndex: int
    totalSlots: int
    occupiedSlots: int
    availableSlots: int
    slots: list[SlotInferenceResult]