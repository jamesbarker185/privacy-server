from pydantic import BaseModel
from typing import Optional

class AnonymizeRequest(BaseModel):
    s3_key: str
    bucket: str
    overwrite: bool = False
    output_prefix: str = "processed/"
    confidence_threshold: float = 0.5

class AnonymizeResponse(BaseModel):
    job_id: str
    status: str
    processed_s3_key: str
    objects_detected: int
