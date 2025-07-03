# -*- coding: utf-8 -*-
"""
Created on Sun Jun  8 13:23:41 2025

@author: shubh
"""

from PIL import Image
import numpy as np

# Load the image
img_name = "pepper"
image_path = f"E://collage_project//project2//images//{img_name}.png"
image = Image.open(image_path).convert('L')  # Load as RGB
np_image = np.array(image)

# Coordinates of the pixel to change (e.g., top-left corner)
x, y = 100, 0  # Change this to any valid (row, col)

# Print original pixel value
print(f"Original pixel value at ({x}, {y}:", np_image[x, y])

# Change the pixel value (e.g., set to pure red)
np_image[x, y] = 0  # RGB: Red

# Print new pixel value
print(f"New pixel value at ({x}, {y}):", np_image[x, y])

# Convert back to image and save
modified_image = Image.fromarray(np_image)
modified_image.save(f"E://collage_project//project2//images//modified_{img_name}.png")
print(f"Modified image saved as 'modified_{img_name}.png'")
