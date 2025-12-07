from pydantic_settings import BaseSettings
from typing import Optional

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Privacy Anonymization Service"
    
    AWS_REGION: str = "us-east-1"
    S3_BUCKET_NAME: str = "drone-raw-data"
    
    # Model Config
    MODEL_CONFIDENCE_THRESHOLD: float = 0.5
    USE_GPU: bool = False
    
    class Config:
        case_sensitive = True
        env_file = ".env"

settings = Settings()
