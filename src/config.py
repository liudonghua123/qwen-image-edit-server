from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional

class Settings(BaseSettings):
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    API_KEY: Optional[str] = None
    
    DEVICE_MAP: str = "auto"
    DEVICE_MAP: str = "auto"
    PRETRAINED_MODEL_NAME: str = "Tongyi-MAI/Z-Image-Turbo"
    MODEL_ID: str = "Qwen/Qwen-Image-Edit-2509" # Used for API response ID
    HF_HOME: Optional[str] = None
    
    # Default Generation Parameters
    DEFAULT_NEGATIVE_PROMPT: str = "blurry, low quality"
    DEFAULT_GUIDANCE_SCALE: float = 7.5
    DEFAULT_TRUE_CFG_SCALE: float = 4.0
    DEFAULT_NUM_INFERENCE_STEPS: int = 40
    DEFAULT_SEED: Optional[int] = None

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

settings = Settings()
