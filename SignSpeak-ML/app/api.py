from fastapi import APIRouter
from typing import List

from .schemas import FrameData
from .model_logic.PipelineManager import PipelineManager
from .model_logic.utils.util_functions import generate_end_of_sentence_response

router = APIRouter()

# Singleton pipeline (model load only once)
pipeline = PipelineManager()


@router.post("/predict_landmarks")
def predict_landmarks(frames: List[FrameData]):
    """
    Accepts a list of FrameData objects.
    Returns detected words or sentences.
    """
    try:
        responses = pipeline.process(frames)
        return {"results": responses}
    except Exception as e:
        return {"error": str(e)}


@router.post("/force_end_sentence")
def force_end_sentence():
    """Force sentence finalization as if SPECIAL_LABEL was predicted."""
    try:
        response = pipeline.force_end_sentence()
        return {"results": [response]}
    except Exception as e:
        return {"error": str(e)}


@router.post("/reset_buffer")
def reset_buffer():
    """Reset internal word buffer. Optionally clears accumulated sentences."""
    try:
        pipeline.reset_buffer()
        return {"status": "ok", "detail": "Buffer reset"}
    except Exception as e:
        return {"error": str(e)}
