import os

from pydantic.v1 import BaseSettings


# config.py




class Settings(BaseSettings):

    # ==== SEGMENTATOR ====
    MOTION_THRESHOLD: float = 0.12
    SILENCE_FRAMES: int = 6
    MIN_WORD_FRAMES: int = 8
    BURST_MULTIPLIER: float = 2.00
    EMA_ALPHA = 0.40

    # ==== DATA PREPARER ====
    EXPECTED_FRAMES: int = 60
    ADD_FEATURES: bool = False

    # ===== MODEL ======
    MODEL_PATH: str = os.getenv('MODEL_PATH', "/app/app/model_logic/utils/model_configs/conf(large).pt")
    DEVICE: str = "cpu"
    POLISHING_MODEL_PATH: str = os.getenv('POLISHING_MODEL_PATH',
                                          "/app/app_models/qwen2.5-1.5b-instruct-q4_k_m.gguf")

    USE_T5:bool = os.getenv('USE_T5', False)

    # ===== OTHER ====
    USE_SEGMENTATOR: bool = os.getenv('USE_SEGMENTATOR', True)
    SPECIAL_LABEL: str = os.getenv('SPECIAL_LABEL', 'PUSH')


settings = Settings()

