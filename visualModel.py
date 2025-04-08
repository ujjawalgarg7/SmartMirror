import matplotlib.pyplot as plt
import cv2

# Specify one of the predicted images (update the path if different)
image_path = "runs/detect/predict/image_171.jpg"
# Read and convert BGR (OpenCV) to RGB for matplotlib
img = cv2.imread(image_path)
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.figure(figsize=(10, 8))
plt.imshow(img_rgb)
plt.title("Detection Output")
plt.axis("off")
plt.show()
