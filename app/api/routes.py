from fastapi import APIRouter, HTTPException, Depends
from app.api.schemas import AnonymizeRequest, AnonymizeResponse
from app.services.job_processor import JobProcessor
import structlog

router = APIRouter()
logger = structlog.get_logger()

@router.post("/anonymize", response_model=AnonymizeResponse)
async def anonymize_image(request: AnonymizeRequest):
    try:
        processor = JobProcessor()
        result = processor.process_job(
            bucket=request.bucket,
            key=request.s3_key,
            overwrite=request.overwrite,
            output_prefix=request.output_prefix
        )
        return result
    except Exception as e:
        logger.error("job_failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
