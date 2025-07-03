# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:55:29 2024

@author: shubh
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def add_gaussian_noise(image, mean=0, std_dev=20):
    """
    Adds Gaussian noise to an image.
    
    :param image: Input image as a NumPy array.
    :param mean: Mean of the Gaussian distribution (default: 0).
    :param std_dev: Standard deviation of the Gaussian noise (default: 20).
    :return: Image with Gaussian noise added.
    """
    noise = np.random.normal(mean, std_dev, image.shape)
    noisy_image = image + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

# Load and encrypt an image (example)
img_name = "pepper"
image_path = f"E://collage_project//project2//images//{img_name}_encrypted_hfpso.png"
encrypted_image = np.array(Image.open(image_path).convert('L'))

# Apply Gaussian noise attack
mean = 0
std_dev = 10  # Adjust this to increase/decrease noise intensity
noisy_image = add_gaussian_noise(encrypted_image, mean, std_dev)

noisy_image_path = f"E://collage_project//project2//images//{img_name}_noisy_image.png"
Image.fromarray(noisy_image).save(noisy_image_path)
print(f"Image saved at: {noisy_image_path}")

# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Encrypted Image")
plt.imshow(encrypted_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("After Gaussian Noise Attack")
plt.imshow(noisy_image, cmap='gray')
plt.axis('off')

plt.show()
