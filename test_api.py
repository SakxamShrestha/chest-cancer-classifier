import requests
from pathlib import Path
import sys

def test_prediction(image_path=None):
    url = "http://0.0.0.0:8000/api/predict"
    
    if image_path is None:
        # Use a default test image from our dataset
        image_path = Path(__file__).parent / "data" / "raw" / "Data" / "test" / "normal" / "normal (1).png"
    else:
        image_path = Path(image_path)
    
    # Check if file exists
    if not image_path.exists():
        print(f"Error: Image file not found at {image_path}")
        print("Available test directories:")
        test_dir = Path(__file__).parent / "data" / "raw" / "Data" / "test"
        for d in test_dir.glob("*"):
            print(f"- {d.name}")
            if d.is_dir():
                for f in d.glob("*.png"):
                    print(f"  - {f.name}")
                    break  # Just show first image as example
        return
    
    print(f"Sending image: {image_path}")
    print(f"File exists: {image_path.exists()}")
    print(f"File size: {image_path.stat().st_size} bytes")
    
    try:
        with open(image_path, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)
            print("\nResponse:")
            print(response.json())
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        test_prediction(image_path)
    else:
        # Run without arguments to use default test image
        test_prediction()