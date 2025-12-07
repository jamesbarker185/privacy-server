import numpy as np
from typing import List
import structlog
import cv2
import os
import torch
from facenet_pytorch import MTCNN

logger = structlog.get_logger()

class InferenceEngine:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(InferenceEngine, cls).__new__(cls)
            cls._instance.initialize()
        return cls._instance

    def initialize(self):
        """Load models here."""
        logger.info("loading_models")
        
        # 1. Face Detection (MTCNN)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.mtcnn = MTCNN(keep_all=True, device=device)
        
        # 2. Plate Detection (Haar Cascade)
        cascade_path = os.path.join(os.path.dirname(__file__), 'data', 'haarcascade_russian_plate_number.xml')
        if not os.path.exists(cascade_path):
            logger.error("cascade_not_found", path=cascade_path)
            self.plate_cascade = None
        else:
            self.plate_cascade = cv2.CascadeClassifier(cascade_path)
            
        self.models_loaded = True

    def detect_faces(self, image: np.array) -> List[List[int]]:
        """
        Returns list of [x, y, w, h] for detected faces.
        """
        if image is None:
            return []
            
        # MTCNN expects RGB
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        try:
            boxes, _ = self.mtcnn.detect(img_rgb)
        except Exception as e:
            logger.error("mtcnn_error", error=str(e))
            return []
            
        result = []
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                result.append([int(x1), int(y1), int(w), int(h)])
                
        return result

    def detect_plates(self, image: np.array) -> List[List[int]]:
        """
        Returns list of [x, y, w, h] for detected license plates.
        """
        if self.plate_cascade is None:
            return []
            
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        plates = self.plate_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        # plates is already a list of [x, y, w, h] or empty tuple
        if len(plates) == 0:
            return []
            
        # Convert numpy array to list
        return [list(p) for p in plates]
