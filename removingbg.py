from rembg import remove
import cv2
import numpy as np
from PIL import Image

# Load input image using PIL
input_path = "./image.jpeg"  # Replace with your image path
image_pil = Image.open(input_path)

# Remove background using rembg
image_no_bg = remove(image_pil)

# Convert to OpenCV format (PIL → NumPy array)
image_cv = np.array(image_no_bg)

# Convert RGB to BGR (since OpenCV uses BGR format)
image_cv = cv2.cvtColor(image_cv, cv2.COLOR_RGB2BGR)

# Apply Gaussian blur for smoother edges (Optional)
blurred = cv2.GaussianBlur(image_cv, (5, 5), 0)

# Save the processed image
output_path = "output.png"
cv2.imwrite(output_path, blurred)

print(f"Background removed and processed image saved as {output_path}")