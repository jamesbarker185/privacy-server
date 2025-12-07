import numpy as np
from app.utils.image_utils import ImageUtils

def test_slice_image():
    # Create a 2048x2048 dummy image
    image = np.zeros((2048, 2048, 3), dtype=np.uint8)
    tiles = ImageUtils.slice_image(image, tile_size=1024, overlap=0.0)
    
    # Should be exactly 4 tiles (2x2)
    assert len(tiles) == 4
    assert tiles[0][1] == 0 # x
    assert tiles[0][2] == 0 # y
    assert tiles[3][1] == 1024 # x
    assert tiles[3][2] == 1024 # y

def test_merge_boxes():
    # Two overlapping boxes
    boxes = [
        [100, 100, 50, 50],
        [105, 105, 45, 45] # Highly overlapping
    ]
    merged = ImageUtils.merge_boxes(boxes, iou_threshold=0.5)
    assert len(merged) == 1
    
    # Two separate boxes
    boxes = [
        [100, 100, 50, 50],
        [300, 300, 50, 50]
    ]
    merged = ImageUtils.merge_boxes(boxes, iou_threshold=0.5)
    assert len(merged) == 2
