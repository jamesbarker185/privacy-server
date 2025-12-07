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

    def process_image_data(self, image: np.array) -> tuple[np.array, list]:
        h, w, _ = image.shape
        logger.info("image_decoded", width=w, height=h)
        
        all_detections = [] # List of (box, label)
        
        # 3. Tiling & Inference
        if w > 2000 or h > 2000:
            logger.info("using_tiling_strategy")
            tiles = ImageUtils.slice_image(image)
            for tile, x_off, y_off in tiles:
                # Detect faces
                face_boxes = self.inference_engine.detect_faces(tile)
                for b in face_boxes:
                    # Adjust coordinates
                    global_box = [b[0] + x_off, b[1] + y_off, b[2], b[3]]
                    all_detections.append((global_box, 'face'))
                    
                # Detect plates
                plate_boxes = self.inference_engine.detect_plates(tile)
                for b in plate_boxes:
                    global_box = [b[0] + x_off, b[1] + y_off, b[2], b[3]]
                    all_detections.append((global_box, 'plate'))
        else:
            face_boxes = self.inference_engine.detect_faces(image)
            for b in face_boxes:
                all_detections.append((b, 'face'))
                
            plate_boxes = self.inference_engine.detect_plates(image)
            for b in plate_boxes:
                all_detections.append((b, 'plate'))
            
        # 4. Merge Boxes (NMS) - simplified for now, just merging boxes regardless of label
        # Ideally we should NMS per class or keep labels. 
        # For now, let's just extract boxes for blurring
        boxes_only = [d[0] for d in all_detections]
        final_boxes = ImageUtils.merge_boxes(boxes_only)
        
        # Re-associate labels (naive approach: if a final box overlaps significantly with a detection, inherit label)
        # For simplicity in this iteration, we will just treat all as 'object' if we use merge_boxes, 
        # OR we skip merge_boxes for metadata if we want precise labels.
        # Let's skip merge_boxes for metadata output to preserve individual detections, 
        # BUT use merged boxes for blurring to avoid double blurring.
        
        metadata = []
        for box, label in all_detections:
            # Normalize coordinates
            nx = box[0] / w
            ny = box[1] / h
            nw = box[2] / w
            nh = box[3] / h
            metadata.append({
                "label": label,
                "box": [nx, ny, nw, nh],
                "score": 1.0 # Placeholder
            })

        logger.info("objects_detected", count=len(final_boxes))
        
        # 5. Blur
        processed_image = self.blurrer.apply_blur(image, final_boxes)
        return processed_image, metadata

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
        processed_image, metadata = self.process_image_data(image)
        
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
            "objects_detected": len(metadata),
            "metadata": metadata
        }

    def process_local_job(self, input_path: str, output_path: str) -> dict:
        job_id = str(uuid.uuid4())
        logger.info("starting_local_job", job_id=job_id, input_path=input_path)
        
        # 1. Read
        image = cv2.imread(input_path)
        if image is None:
            raise ValueError(f"Failed to read image from {input_path}")
            
        # 2. Process
        processed_image, metadata = self.process_image_data(image)
        
        # 3. Write Image
        success = cv2.imwrite(output_path, processed_image)
        if not success:
            raise IOError(f"Failed to write image to {output_path}")
            
        return {
            "job_id": job_id,
            "status": "success",
            "output_path": output_path,
            "objects_detected": len(metadata),
            "metadata": metadata
        }
