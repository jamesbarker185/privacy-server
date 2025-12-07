import numpy as np
from typing import List, Tuple

class ImageUtils:
    @staticmethod
    def slice_image(image: np.array, tile_size: int = 1024, overlap: float = 0.2) -> List[Tuple[np.array, int, int]]:
        """
        Slices an image into overlapping tiles.
        Returns a list of (tile_image, x_offset, y_offset).
        """
        h, w, _ = image.shape
        step = int(tile_size * (1 - overlap))
        
        tiles = []
        for y in range(0, h, step):
            for x in range(0, w, step):
                # Calculate tile boundaries
                y_end = min(y + tile_size, h)
                x_end = min(x + tile_size, w)
                
                # Adjust start if we're at the edge to ensure full tile size if possible
                # (Optional optimization, for now simple slicing is fine)
                
                tile = image[y:y_end, x:x_end]
                tiles.append((tile, x, y))
                
        return tiles

    @staticmethod
    def merge_boxes(boxes: List[List[int]], iou_threshold: float = 0.5) -> List[List[int]]:
        """
        Applies Non-Maximum Suppression (NMS) to merge overlapping boxes from different tiles.
        Box format: [x, y, w, h]
        """
        if not boxes:
            return []

        # Convert to [x1, y1, x2, y2] for NMS
        boxes_xyxy = []
        for b in boxes:
            boxes_xyxy.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])
            
        boxes_xyxy = np.array(boxes_xyxy)
        
        # Simple NMS implementation
        pick = []
        x1 = boxes_xyxy[:, 0]
        y1 = boxes_xyxy[:, 1]
        x2 = boxes_xyxy[:, 2]
        y2 = boxes_xyxy[:, 3]
        
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        idxs = np.argsort(y2)
        
        while len(idxs) > 0:
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            overlap = (w * h) / area[idxs[:last]]
            
            idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > iou_threshold)[0])))
            
        # Convert back to [x, y, w, h]
        final_boxes = []
        for i in pick:
            b = boxes_xyxy[i]
            final_boxes.append([int(b[0]), int(b[1]), int(b[2] - b[0]), int(b[3] - b[1])])
            
        return final_boxes
