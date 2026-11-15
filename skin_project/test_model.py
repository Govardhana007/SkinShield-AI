import requests
from PIL import Image
import numpy as np
from io import BytesIO

# Create a test image (random RGB image, 224x224)
np.random.seed(42)
img_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
test_image = Image.fromarray(img_array, 'RGB')

img_bytes = BytesIO()
test_image.save(img_bytes, format='PNG')
img_bytes.seek(0)

# Send to prediction endpoint
files = {'image': ('test.png', img_bytes, 'image/png')}
response = requests.post('http://localhost:5000/predict', files=files)

print("=" * 50)
print("TEST RESULTS")
print("=" * 50)
print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
print("=" * 50)
