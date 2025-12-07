import os
import sys
import glob
import structlog

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app.services.job_processor import JobProcessor

logger = structlog.get_logger()

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
    
    results = []
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        output_path = os.path.join(output_dir, f"processed_{filename}")
        
        print(f"Processing {filename}...")
        try:
            result = processor.process_local_job(img_path, output_path)
            count = result['objects_detected']
            print(f"Success: {filename} -> {count} objects detected.")
            results.append((filename, count, "Success"))
        except Exception as e:
            print(f"Failed to process {filename}: {e}")
            logger.exception("processing_failed", filename=filename)
            results.append((filename, 0, f"Failed: {e}"))

    print("\n" + "="*50)
    print(f"{'Filename':<30} | {'Objects':<10} | {'Status':<10}")
    print("-" * 50)
    for filename, count, status in results:
        print(f"{filename:<30} | {count:<10} | {status:<10}")
    print("="*50)

if __name__ == "__main__":
    run_tests()
