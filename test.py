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

def func(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  
    resized_image = cv2.resize(image, (100, 100))  

    pixels = resized_image.reshape((-1, 3))

    n_colors = 5  
    gmm = GaussianMixture(n_components=n_colors, random_state=0)
    gmm.fit(pixels)

    cluster_colors = gmm.means_.astype(int)
    cluster_weights = gmm.weights_

    dominant_index = np.argmax(cluster_weights)
    dominant_color = cluster_colors[dominant_index]
    hex_code = rgb_to_hex(dominant_color)

    print(f"Dominant Color (RGB): {dominant_color}")
    print(f"Dominant Color (Hex): {hex_code}")


    plt.figure(figsize=(2, 2))
    plt.imshow([[dominant_color / 255]]) 
    plt.title(f"Hex {hex_code}", fontsize=10)
    plt.axis('off')
    plt.show()


    color_rgb = (int(dominant_color[0]),int(dominant_color[1]),int(dominant_color[2]))

    data = pd.read_csv('color_data.csv')

    color_dict = data.groupby('Color_Family').apply(lambda group: list(group[['R', 'G', 'B']].itertuples(index=False, name=None))).to_dict()


    k = 3

    print("\n\ncolor is : {}".format(classifyAPoint(color_dict, color_rgb, k)))



cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Could not open camera.")
else:
    print("Press 'c' to capture an image or 'q' to quit.")

    captured_frame = None
    
    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to capture frame.")
            break


        cv2.imshow('Camera Preview', frame)


        key = cv2.waitKey(1) & 0xFF

        if key == ord('c'):  
            captured_frame = frame
            print("Image captured and stored in variable.")
            print(type(captured_frame))
            func(captured_frame)
            
        elif key == ord('q'):  
            print("Exiting without capturing.")
            break


    cap.release()
    cv2.destroyAllWindows()


    if captured_frame is not None:
        print("Captured frame is now available in the variable.")

