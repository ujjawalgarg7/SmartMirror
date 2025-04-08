from kivy.app import App
from kivy.uix.relativelayout import RelativeLayout
from kivy.uix.label import Label
from kivy.uix.image import Image
from kivy.clock import Clock
import cv2
import requests
from datetime import datetime
from kivy.graphics.texture import Texture
from kivy.uix.boxlayout import BoxLayout
from kivy.graphics import Color, RoundedRectangle
from dotenv import load_dotenv
import os
import math
from collections import defaultdict
from sklearn.mixture import GaussianMixture
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from kivy.core.window import Window

load_dotenv()
# Weather API Setup
CITY = "Noida"
API_KEY = os.getenv("API_KEY")  # Replace with actual API key


def classifyAPoint(points, p, k=3):
    distance = []

    for group in points:
        for feature in points[group]:
            euclidean_distance = math.sqrt((feature[0] - p[0]) ** 2 + (feature[1] - p[1]) ** 2 + (feature[2] - p[2]) ** 2)
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

    color_rgb = (int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2]))

    data = pd.read_csv('color_data.csv')

    color_dict = data.groupby('Color_Family').apply(
        lambda group: list(group[['R', 'G', 'B']].itertuples(index=False, name=None))).to_dict()

    k = 3

    print("\n\ncolor is : {}".format(classifyAPoint(color_dict, color_rgb, k)))


class SmartMirror(RelativeLayout):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        # OpenCV Camera Capture
        self.capture = cv2.VideoCapture(0)

        # Fullscreen Camera Feed
        self.camera_feed = Image(size_hint=(1, 1), allow_stretch=True, keep_ratio=False)
        self.add_widget(self.camera_feed)

        # Bottom-Left Text Container (Horizontal)
        self.text_container = BoxLayout(orientation='horizontal', size_hint=(None, 0.3),  # Dynamic width
                                         pos_hint={"left": 0, "bottom": 0}, padding=[20, 20, 20, 20],
                                         spacing=20)  # Initial spacing

        # Create a Smooth Semi-Transparent Background Behind Text
        with self.canvas.before:
            Color(0, 0, 0, 1)  # 50% Black Transparency
            self.overlay_rect = RoundedRectangle(pos=self.text_container.pos,
                                                 size=self.text_container.size,
                                                 radius=[10])  # reduced radius

        self.add_widget(self.text_container)

        # Name Label (Elegant & Modern)
        self.name_label = Label(text="Ujjawal Garg", font_size=40, bold=True,
                                 color=(1, 0, 0, 1), font_name="Arial", halign="left",
                                 text_size=(None, self.text_container.height))  # reduced font size and changed halign
        self.text_container.add_widget(self.name_label)

        # Date & Time Label (More Spacing & Readable)
        self.datetime_label = Label(font_size=30, bold=True, color=(1, 1, 0, 1),
                                     font_name="Arial", halign="left",
                                     text_size=(None, self.text_container.height))  # reduced font size and changed halign
        self.text_container.add_widget(self.datetime_label)

        # Weather Label Container (RelativeLayout)
        self.weather_container = RelativeLayout(size_hint=(None, None),
                                                 pos_hint={"right": 1, "bottom": 0})

        self.add_widget(self.weather_container)

        # Weather Label (Modern Font & Icon) - Inside the container
        self.weather_label = Label(font_size=30, bold=True, color=(0, 0, 0, 1),
                                     font_name="Arial", halign="right",
                                     size_hint=(None, 1),
                                     pos_hint={"right": 0.2, "bottom": 0},
                                     text_size=(250, None),
                                     padding=[0, 0, 40, 80])  # Added padding here

        self.weather_container.add_widget(self.weather_label)

        # Smart Mirror Label (Top Center)
        self.smart_mirror_label = Label(text="SMART MIRROR", font_size=50, bold=True,
                                         color=(0.2, 0.5, 1, 1), font_name="Arial",
                                         pos_hint={"top": 1, "center_x": 0.5, "center_y": 0.95})
        self.add_widget(self.smart_mirror_label)

        # Update Time, Weather, and Camera Feed
        Clock.schedule_interval(self.update_time, 1)
        Clock.schedule_interval(self.update_weather, 600)
        Clock.schedule_interval(self.update_camera, 1.0 / 30)
        self.update_weather()

        # Ensure Background Updates and Text Container Size
        Clock.schedule_interval(self.update_text_container_size, 0.1)
        self.bind(size=self.update_overlay, pos=self.update_overlay)

        # Keyboard event handling
        Window.bind(on_key_down=self.on_keyboard_down)

    def on_keyboard_down(self, instance, keyboard, keycode, text, modifiers):
        if text == 'c':
            ret, frame = self.capture.read()
            if ret:
                func(frame)

    def update_text_container_size(self, dt):
        """Update text container size dynamically."""
        labels = [self.name_label, self.datetime_label]  # Exclude weather_label
        total_text_width = sum([label.texture_size[0] for label in labels])
        text_width_padding = 50  # add extra pixels for padding
        self.text_container.size_hint = (None, 0.3)
        self.text_container.width = total_text_width + text_width_padding + (
                    len(labels) - 1) * self.text_container.spacing

    def update_overlay(self, *args):
        """ Update the semi-transparent background size & position dynamically """
        self.overlay_rect.size = self.text_container.size