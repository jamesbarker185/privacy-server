import boto3
import io
from botocore.exceptions import ClientError
from app.core.config import settings
import structlog

logger = structlog.get_logger()

class S3Handler:
    def __init__(self):
        self.s3_client = boto3.client('s3', region_name=settings.AWS_REGION)

    def download_image(self, bucket: str, key: str) -> bytes:
        """Downloads an image from S3 and returns it as bytes."""
        try:
            logger.info("downloading_image", bucket=bucket, key=key)
            response = self.s3_client.get_object(Bucket=bucket, Key=key)
            return response['Body'].read()
        except ClientError as e:
            logger.error("s3_download_failed", error=str(e), bucket=bucket, key=key)
            raise e

    def upload_image(self, image_bytes: bytes, bucket: str, key: str):
        """Uploads an image (bytes) to S3."""
        try:
            logger.info("uploading_image", bucket=bucket, key=key)
            self.s3_client.put_object(
                Bucket=bucket,
                Key=key,
                Body=image_bytes,
                ContentType='image/jpeg'
            )
        except ClientError as e:
            logger.error("s3_upload_failed", error=str(e), bucket=bucket, key=key)
            raise e
