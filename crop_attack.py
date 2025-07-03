# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 17:47:49 2024

@author: shubh
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def crop_attack(encrypted_image, crop_size):
    """
    Simulates a crop attack by removing a rectangular region from the encrypted image.
    
    :param encrypted_image: Encrypted image as a NumPy array.
    :param crop_size: Tuple (crop_height, crop_width) indicating the size of the cropped region.
    :return: Attacked image with the cropped region set to zero.
    """
    attacked_image = encrypted_image.copy()
    M, N = encrypted_image.shape

    # Define the cropping region (centered for simplicity)
    crop_height, crop_width = crop_size
    # start_row = (M - crop_height) // 2
    # start_col = (N - crop_width) // 2
    start_row = 0
    start_col = 0

    # Set the cropped region to zero
    attacked_image[start_row:start_row + crop_height, start_col:start_col + crop_width] = 0
    return attacked_image

# Load and encrypt an image
img_name = "pepper"
image_path = f"E://collage_project//project2//images//{img_name}_encrypted_hfpso.png"
encrypted_image = np.array(Image.open(image_path).convert('L'))

# Apply crop attack
crop_size = (128,128)
attacked_image = crop_attack(encrypted_image, crop_size)

attacked_image_path = f"E://collage_project//project2//images//{img_name}_cropped_image.png"
Image.fromarray(attacked_image).save(attacked_image_path)
print(f"Image saved at: {attacked_image_path}")
# Display the results
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title("Encrypted Image")
plt.imshow(encrypted_image, cmap='gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title("After Crop Attack")
plt.imshow(attacked_image, cmap='gray')
plt.axis('off')

plt.show()
