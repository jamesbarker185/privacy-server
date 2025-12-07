import cv2
import numpy as np
from typing import List

class PrivacyBlurrer:
    def apply_blur(self, image: np.array, boxes: List[List[int]]) -> np.array:
        """
        Applies Gaussian Blur to the regions specified by boxes.
        boxes: List of [x, y, w, h]
        """
        processed_image = image.copy()
        
        for (x, y, w, h) in boxes:
            # Ensure coordinates are within image bounds
            h_img, w_img, _ = processed_image.shape
            x = max(0, x)
            y = max(0, y)
            w = min(w, w_img - x)
            h = min(h, h_img - y)
            
            if w <= 0 or h <= 0:
                continue
                
            roi = processed_image[y:y+h, x:x+w]
            
            # Dynamic kernel size based on ROI size
            k_w = (w // 3) | 1 # Ensure odd
            k_h = (h // 3) | 1 # Ensure odd
            
            # Cap kernel size to avoid errors if roi is too small
            k_w = max(3, k_w)
            k_h = max(3, k_h)
            
            blurred_roi = cv2.GaussianBlur(roi, (k_w, k_h), 0)
            processed_image[y:y+h, x:x+w] = blurred_roi
            
        return processed_image
