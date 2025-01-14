import cv2
import numpy as np
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import math
from collections import defaultdict
import pandas as pd


def classifyAPoint(points, p, k=3):
    distance = []

    for group in points:
        for feature in points[group]:
            euclidean_distance = math.sqrt((feature[0]-p[0])**2 + (feature[1]-p[1])**2 + (feature[2]-p[2])**2)
            distance.append((euclidean_distance, group))

    distance = sorted(distance, key=lambda x: x[0])[:k]

    freq = defaultdict(int)
    for d in distance:
        freq[d[1]] += 1

    return max(freq, key=freq.get)


def rgb_to_hex(rgb):
    """Convert an RGB color to a hexadecimal string."""
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

# Load and preprocess the image
image_path = './purple.jpg'  # Replace with your image path
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
resized_image = cv2.resize(image, (100, 100))  # Resize for faster processing

# Reshape the image into a 2D array of RGB values
pixels = resized_image.reshape((-1, 3))

# Apply Gaussian Mixture Model
n_colors = 5  # Number of clusters to identify
gmm = GaussianMixture(n_components=n_colors, random_state=0)
gmm.fit(pixels)

# Extract cluster means (dominant colors) and their weights
cluster_colors = gmm.means_.astype(int)
cluster_weights = gmm.weights_

# Identify the dominant color
dominant_index = np.argmax(cluster_weights)
dominant_color = cluster_colors[dominant_index]
hex_code = rgb_to_hex(dominant_color)

# Output the dominant color in RGB and Hex
print(f"Dominant Color (RGB): {dominant_color}")
print(f"Dominant Color (Hex): {hex_code}")

color_rgb = (int(dominant_color[0]),int(dominant_color[1]),int(dominant_color[2]))

data = pd.read_csv('color_data.csv')

color_dict = data.groupby('Color_Family').apply(lambda group: list(group[['R', 'G', 'B']].itertuples(index=False, name=None))).to_dict()

#p = (204,11,14)

k = 3

print("The value classified to unknown point is: {}".format(classifyAPoint(color_dict, color_rgb, k)))