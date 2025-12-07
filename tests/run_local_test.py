import os
import sys
import glob
import structlog
import time
import csv
import cv2
import json
import shutil

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.job_processor import JobProcessor

logger = structlog.get_logger()

def generate_viewer(output_dir, image_files):
    html_content = """
<!DOCTYPE html>
<html>
<head>
    <title>Privacy Service Viewer</title>
    <style>
        body { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; padding: 20px; background: #1a1a1a; color: #e0e0e0; }
        .container { display: flex; flex-wrap: wrap; gap: 20px; justify-content: center; }
        .image-card { background: #2d2d2d; padding: 10px; border-radius: 8px; box-shadow: 0 4px 6px rgba(0,0,0,0.3); position: relative; }
        .image-wrapper { position: relative; display: inline-block; overflow: hidden; border-radius: 4px; }
        img { max-width: 800px; display: block; }
        
        .blur-mask { 
            position: absolute; 
            backdrop-filter: blur(15px); 
            -webkit-backdrop-filter: blur(15px);
            background-color: rgba(255, 255, 255, 0.1); /* Slight tint to make it visible if blur fails or on flat colors */
            pointer-events: none; 
            z-index: 10;
            transition: opacity 0.3s ease;
        }
        
        .controls { margin-bottom: 20px; padding: 20px; background: #2d2d2d; border-radius: 8px; text-align: center; }
        h1 { margin-top: 0; }
        
        /* Toggle Switch */
        .switch { position: relative; display: inline-block; width: 60px; height: 34px; vertical-align: middle; }
        .switch input { opacity: 0; width: 0; height: 0; }
        .slider { position: absolute; cursor: pointer; top: 0; left: 0; right: 0; bottom: 0; background-color: #ccc; transition: .4s; border-radius: 34px; }
        .slider:before { position: absolute; content: ""; height: 26px; width: 26px; left: 4px; bottom: 4px; background-color: white; transition: .4s; border-radius: 50%; }
        input:checked + .slider { background-color: #2196F3; }
        input:checked + .slider:before { transform: translateX(26px); }
        .toggle-label { margin-left: 10px; font-size: 18px; vertical-align: middle; }
    </style>
</head>
<body>
    <div class="controls">
        <h1>Privacy Service Results</h1>
        <label class="switch">
            <input type="checkbox" id="toggle-masks" checked>
            <span class="slider"></span>
        </label>
        <span class="toggle-label">Censorship Active</span>
    </div>
    <div class="container" id="gallery"></div>

    <script>
        const images = [
            IMAGE_LIST_PLACEHOLDER
        ];

        const gallery = document.getElementById('gallery');
        const toggle = document.getElementById('toggle-masks');

        images.forEach(imgData => {
            const card = document.createElement('div');
            card.className = 'image-card';
            
            const wrapper = document.createElement('div');
            wrapper.className = 'image-wrapper';
            
            const img = document.createElement('img');
            img.src = imgData.original_filename; // Use original image
            wrapper.appendChild(img);
            
            imgData.metadata.forEach(obj => {
                const box = document.createElement('div');
                box.className = 'blur-mask';
                // box is [x, y, w, h] normalized
                box.style.left = (obj.box[0] * 100) + '%';
                box.style.top = (obj.box[1] * 100) + '%';
                box.style.width = (obj.box[2] * 100) + '%';
                box.style.height = (obj.box[3] * 100) + '%';
                
                // Optional: Tooltip or label on hover?
                box.title = obj.label;
                
                wrapper.appendChild(box);
            });
            
            card.appendChild(wrapper);
            
            const info = document.createElement('div');
            info.innerHTML = `<div style="margin-top:10px"><strong>${imgData.filename}</strong><br>Objects: ${imgData.metadata.length}</div>`;
            card.appendChild(info);
            
            gallery.appendChild(card);
        });

        toggle.addEventListener('change', (e) => {
            const boxes = document.querySelectorAll('.blur-mask');
            boxes.forEach(b => b.style.opacity = e.target.checked ? '1' : '0');
        });
    </script>
</body>
</html>
    """
    
    # Prepare data for JS
    js_data = []
    for img_file in image_files:
        filename = os.path.basename(img_file)
        original_filename = f"original_{filename}"
        json_filename = f"processed_{filename}.json"
        
        json_path = os.path.join(output_dir, json_filename)
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                metadata = json.load(f)
                js_data.append({
                    'filename': filename,
                    'original_filename': original_filename,
                    'metadata': metadata
                })
    
    final_html = html_content.replace('IMAGE_LIST_PLACEHOLDER', json.dumps(js_data)[1:-1]) # Remove outer brackets to splice in
    
    with open(os.path.join(output_dir, 'viewer.html'), 'w') as f:
        f.write(final_html)
    print(f"Viewer generated at {os.path.join(output_dir, 'viewer.html')}")

def run_tests():
    input_dir = os.path.join(os.path.dirname(__file__), 'sampleData')
    output_dir = os.path.join(os.path.dirname(__file__), 'output')
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    processor = JobProcessor()
    
    # Find all png files
    image_files = glob.glob(os.path.join(input_dir, "*.png"))
    
    if not image_files:
        print(f"No .png files found in {input_dir}")
        return

    print(f"Found {len(image_files)} images to process.")
    
    stats = []
    csv_file = os.path.join(output_dir, 'stats.csv')
    total_start_time = time.time()
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        json_path = output_path + ".json"
        original_copy_path = os.path.join(output_dir, f"original_{filename}")
        
        # Copy original image for viewer
        shutil.copy(img_path, original_copy_path)
        
        # Get image size for stats
        img = cv2.imread(img_path)
        if img is None:
            h, w = 0, 0
        else:
            h, w, _ = img.shape
        
        print(f"Processing {filename} ({w}x{h})...")
        
        start_time = time.time()
        try:
            result = processor.process_local_job(img_path, output_path)
            duration = time.time() - start_time
            
            count = result['objects_detected']
            metadata = result.get('metadata', [])
            
            # Save metadata
            with open(json_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            print(f"Success: {filename} -> {count} objects detected in {duration:.2f}s")
            
            stats.append({
                'filename': filename,
                'width': w,
                'height': h,
                'objects_detected': count,
                'processing_time_seconds': round(duration, 4),
                'status': 'Success',
                'error': ''
            })
        except Exception as e:
            duration = time.time() - start_time
            print(f"Failed to process {filename}: {e}")
            logger.exception("processing_failed", filename=filename)
            
            stats.append({
                'filename': filename,
                'width': w,
                'height': h,
                'objects_detected': 0,
                'processing_time_seconds': round(duration, 4),
                'status': 'Failed',
                'error': str(e)
            })

    total_duration = time.time() - total_start_time
    
    # Write CSV
    with open(csv_file, 'w', newline='') as f:
        fieldnames = ['filename', 'width', 'height', 'objects_detected', 'processing_time_seconds', 'status', 'error']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(stats)
        
    print(f"\nTotal processing time: {total_duration:.2f}s")
    print(f"Stats saved to {csv_file}")
    
    generate_viewer(output_dir, image_files)

    print("\n" + "="*90)
    print(f"{'Filename':<30} | {'Size':<15} | {'Objects':<10} | {'Time (s)':<10} | {'Status':<10}")
    print("-" * 90)
    for s in stats:
        size_str = f"{s['width']}x{s['height']}"
        print(f"{s['filename']:<30} | {size_str:<15} | {s['objects_detected']:<10} | {s['processing_time_seconds']:<10.4f} | {s['status']:<10}")
    print("="*90)

if __name__ == "__main__":
    run_tests()
