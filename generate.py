import cv2
import numpy as np
import os
import random
from math import cos, sin, pi

# Configuration
IMG_SIZE = 416
NUM_IMAGES = 10000
SAVE_DIR = "shapes_dataset"
SHAPES = ["circle", "triangle", "square", "rectangle", "star"]
CLASS_MAP = {name: idx for idx, name in enumerate(SHAPES)}

# Directories
os.makedirs(f"{SAVE_DIR}/images", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/labels", exist_ok=True)

def add_noise(img):
    # Add Gaussian noise
    noise = np.random.normal(0, 25, img.shape).astype(np.uint8)
    img = cv2.add(img, noise)

    # Add random lines
    for _ in range(random.randint(5, 15)):
        pt1 = (random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE))
        pt2 = (random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE))
        color = [random.randint(0, 255) for _ in range(3)]
        cv2.line(img, pt1, pt2, color, 1)

    # Add random dots
    for _ in range(random.randint(50, 100)):
        pt = (random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE))
        img[pt[1] % IMG_SIZE, pt[0] % IMG_SIZE] = [random.randint(0, 255) for _ in range(3)]

    return img

def rotate_points(pts, center, angle_rad):
    rot_pts = []
    for pt in pts:
        x, y = pt[0] - center[0], pt[1] - center[1]
        new_x = x * cos(angle_rad) - y * sin(angle_rad)
        new_y = x * sin(angle_rad) + y * cos(angle_rad)
        rot_pts.append([int(center[0] + new_x), int(center[1] + new_y)])
    return rot_pts

def draw_shape(img, shape_name):
    center = (random.randint(80, IMG_SIZE - 80), random.randint(80, IMG_SIZE - 80))
    size = random.randint(30, 90)
    color = [random.randint(0, 255) for _ in range(3)]
    angle = random.uniform(0, 2 * pi)
    thickness = -1

    if shape_name == "circle":
        cv2.circle(img, center, size, color, thickness)
        return (center[0] - size, center[1] - size, center[0] + size, center[1] + size)

    elif shape_name == "square":
        pts = np.array([
            [center[0] - size, center[1] - size],
            [center[0] + size, center[1] - size],
            [center[0] + size, center[1] + size],
            [center[0] - size, center[1] + size]
        ])
        pts = rotate_points(pts, center, angle)
        pts = np.array(pts).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color)
        x, y, w, h = cv2.boundingRect(pts)
        return (x, y, x + w, y + h)

    elif shape_name == "rectangle":
        w = size
        h = size // 2
        pts = np.array([
            [center[0] - w, center[1] - h],
            [center[0] + w, center[1] - h],
            [center[0] + w, center[1] + h],
            [center[0] - w, center[1] + h]
        ])
        pts = rotate_points(pts, center, angle)
        pts = np.array(pts).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color)
        x, y, w, h = cv2.boundingRect(pts)
        return (x, y, x + w, y + h)

    elif shape_name == "triangle":
        pts = np.array([
            [center[0], center[1] - size],
            [center[0] - size, center[1] + size],
            [center[0] + size, center[1] + size]
        ])
        pts = rotate_points(pts, center, angle)
        pts = np.array(pts).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color)
        x, y, w, h = cv2.boundingRect(pts)
        return (x, y, x + w, y + h)

    elif shape_name == "star":
        points = []
        for i in range(10):
            r = size if i % 2 == 0 else size // 2
            theta = pi / 5 * i + angle
            x = int(center[0] + r * cos(theta))
            y = int(center[1] + r * sin(theta))
            points.append([x, y])
        pts = np.array(points).reshape((-1, 1, 2))
        cv2.fillPoly(img, [pts], color)
        x, y, w, h = cv2.boundingRect(pts)
        return (x, y, x + w, y + h)

def convert_to_yolo_format(bbox, img_w, img_h):
    x1, y1, x2, y2 = bbox
    x_center = ((x1 + x2) / 2) / img_w
    y_center = ((y1 + y2) / 2) / img_h
    width = (x2 - x1) / img_w
    height = (y2 - y1) / img_h
    return x_center, y_center, width, height

# Generate dataset
for i in range(NUM_IMAGES):
    img = np.ones((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8) * 255
    shape = random.choice(SHAPES)
    bbox = draw_shape(img, shape)
    x, y, w, h = convert_to_yolo_format(bbox, IMG_SIZE, IMG_SIZE)
    class_id = CLASS_MAP[shape]

    # Add noise after drawing shape
    img = add_noise(img)

    img_path = f"{SAVE_DIR}/images/image_{i}.jpg"
    label_path = f"{SAVE_DIR}/labels/image_{i}.txt"

    cv2.imwrite(img_path, img)
    with open(label_path, "w") as f:
        f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

    if i % 500 == 0:
        print(f"[+] {i}/{NUM_IMAGES} generated...")

print(f"[✓] Completed: {NUM_IMAGES} images with random shape, size, noise, and orientation saved to '{SAVE_DIR}' ✅")
