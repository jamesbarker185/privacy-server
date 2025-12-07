import cv2
import numpy as np
import uuid
import structlog
from app.services.s3_handler import S3Handler
from app.models.inference_engine import InferenceEngine
from app.services.privacy_blurrer import PrivacyBlurrer
from app.utils.image_utils import ImageUtils

logger = structlog.get_logger()

class JobProcessor:
    def __init__(self):
        self.s3_handler = S3Handler()
        self.inference_engine = InferenceEngine()
        self.blurrer = PrivacyBlurrer()

    def process_image_data(self, image: np.array) -> np.array:
        h, w, _ = image.shape
        logger.info("image_decoded", width=w, height=h)
        
        # 3. Tiling & Inference
        # If image is large (> 2000x2000), use tiling
        all_boxes = []
        if w > 2000 or h > 2000:
            logger.info("using_tiling_strategy")
            tiles = ImageUtils.slice_image(image)
            for tile, x_off, y_off in tiles:
                # Detect faces
                face_boxes = self.inference_engine.detect_faces(tile)
                # Map back to global coordinates
                for b in face_boxes:
                    all_boxes.append([b[0] + x_off, b[1] + y_off, b[2], b[3]])
                    
                # Detect plates
                plate_boxes = self.inference_engine.detect_plates(tile)
                for b in plate_boxes:
                    all_boxes.append([b[0] + x_off, b[1] + y_off, b[2], b[3]])
        else:
            all_boxes.extend(self.inference_engine.detect_faces(image))
            all_boxes.extend(self.inference_engine.detect_plates(image))
            
        # 4. Merge Boxes (NMS)
        final_boxes = ImageUtils.merge_boxes(all_boxes)
        logger.info("objects_detected", count=len(final_boxes))
        
        # 5. Blur
        processed_image = self.blurrer.apply_blur(image, final_boxes)
        return processed_image, len(final_boxes)

    def process_job(self, bucket: str, key: str, overwrite: bool = False, output_prefix: str = "processed/") -> dict:
        job_id = str(uuid.uuid4())
        logger.info("starting_job", job_id=job_id, bucket=bucket, key=key)
        
        # 1. Download
        image_bytes = self.s3_handler.download_image(bucket, key)
        
        # 2. Decode
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Failed to decode image")
            
        # 3-5. Process
        processed_image, object_count = self.process_image_data(image)
        
        # 6. Encode & Upload
        _, encoded_img = cv2.imencode('.jpg', processed_image)
        processed_bytes = encoded_img.tobytes()
        
        if overwrite:
            output_key = key
        else:
            filename = key.split('/')[-1]
            output_key = f"{output_prefix}{filename.replace('.jpg', '_anonymized.jpg')}"
            
        self.s3_handler.upload_image(processed_bytes, bucket, output_key)
        
        return {
            "job_id": job_id,
            "status": "success",
            "processed_s3_key": output_key,
            "objects_detected": object_count
        }

    def process_local_job(self, input_path: str, output_path: str) -> dict:
        job_id = str(uuid.uuid4())
        logger.info("starting_local_job", job_id=job_id, input_path=input_path)
        
        # 1. Read
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Failed to read image from {input_path}")
            
        # 2. Process
        processed_image, object_count = self.process_image_data(image)
        
        # 3. Write
        success = cv2.imwrite(output_path, processed_image)
        if not success:
            raise IOError(f"Failed to write image to {output_path}")
            
        return {
            "job_id": job_id,
            "status": "success",
            "output_path": output_path,
            "objects_detected": object_count
        }
