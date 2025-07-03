# -*- coding: utf-8 -*-
"""
Created on Mon Nov 25 19:28:54 2024

@author: shubh
"""

import numpy as np
import random
from PIL import Image


def logistic_map_sequence(length, x0=0.5, r=3.99):
    sequence = []
    x = x0
    x = x0
    for i in range(10000):
        x = r * x * (1 - x) 
    for _ in range(length):
        x = r * x * (1 - x)
        sequence.append(x)
    # Normalize to integers for pixel positions
    indices = np.argsort(sequence)
    return indices

def generate_key(np_image, num):
    M, N = np_image.shape[:2]
    # Calculate the starting pixel of the 5-pixel block
    row = num // N
    col = num % N
    
    # Take a block of 5 consecutive pixels in the horizontal direction
    # If you prefer vertical, change the indexing accordingly.
    block = []
    for i in range(5):
        # Ensure we do not go out of bounds (handle edge cases)
        if col + i < N:
            block.append(np_image[row, col + i])
        else:
            block.append(np_image[(row+1)%M, (col + i) % N])  # Wrap around to the next row if out of bounds
    
    # Calculate key based on the block of 5 pixels (sum and normalize)
    key = sum(block) / (256 * len(block))
    return key

def logistic_map_key_sequence(size, r, x0):
    key_sequence = np.zeros(size, dtype=np.uint8)
    x = x0
    for i in range(10000):
        x = r * x * (1 - x)  # Logistic map formula
    for i in range(size):
        x = r * x * (1 - x)  # Logistic map formula
        key_sequence[i] = int(x * 256) % 256  # Map to [0, 255]
    return key_sequence

def reverse_backward_diffusion_gray(image_F, key_sequence):
    image_F_1d = image_F.reshape(-1)
    image_B = image_F_1d.copy()
    
    # Apply reverse backward diffusion
    lastInd = len(image_F_1d) - 1
    image_B[lastInd] = 0 ^ key_sequence[lastInd] ^ image_F_1d[lastInd]
    for i in range(len(image_F_1d) - 2, -1, -1):
        image_B[i] = image_F_1d[i+1] ^ key_sequence[i] ^ image_F_1d[i]
    
    return image_B.reshape(image_F.shape)

def reverse_forward_diffusion_gray(image, key_sequence):
    image_1d = image.reshape(-1)
    image_F = image_1d.copy()
    
    # Apply reverse forward diffusion
    image_F[0] = 0 ^ key_sequence[0] ^ image_1d[0]
    for i in range(1, len(image_1d)):
        image_F[i] = image_1d[i] ^ key_sequence[i] ^ image_1d[i-1]
    
    return image_F.reshape(image.shape)

def decrypt_image(np_image, x0, r=3.8):
    size = np_image.size
    key_sequence = logistic_map_key_sequence(size, r, x0)

    dec_shuffle_img = unshuffle_pixels(np_image, x0=x0, r=r)
    rev_back_diffused_image = reverse_backward_diffusion_gray(dec_shuffle_img, key_sequence)
    rev_ford_diffused_image = reverse_forward_diffusion_gray(rev_back_diffused_image, key_sequence)
    
    return rev_ford_diffused_image

def unshuffle_pixels(shuffled_image, x0=0.5, r=3.99):
    flat_image = shuffled_image.flatten()
    total_pixels = flat_image.size
    
    # Generate the logistic map sequence for unshuffling
    shuffle_indices = logistic_map_sequence(total_pixels, x0=x0, r=r)
    
    # Reverse the shuffling by placing pixels back in original positions
    unshuffled_flat_image = np.zeros_like(flat_image)
    unshuffled_flat_image[shuffle_indices] = flat_image
    
    # Reshape back to the original image dimensions
    unshuffled_image = unshuffled_flat_image.reshape(shuffled_image.shape)
    return unshuffled_image

img_name = "pepper"
img_path = f"E://collage_project//project2//images//{img_name}.png"
image = Image.open(img_path)
image = image.convert('L')
np_image = np.array(image,dtype=np.uint8)
best_X0 = 247808
dec_key = generate_key(np_image,int(best_X0))

# crop attack analysis
crop_image_path = f"E://collage_project//project2//images//{img_name}_cropped_image.png"
crop_image = Image.open(crop_image_path)
crop_image = crop_image.convert('L')
crop_np_image = np.array(crop_image,dtype=np.uint8)

dec_img = decrypt_image(crop_np_image,dec_key)
decrypted_image_path = f"E://collage_project//project2//images//{img_name}_decrypted_cropped_image.png"
Image.fromarray(dec_img).save(decrypted_image_path)
print(f"Image saved at: {decrypted_image_path}")

# noise attack analysis
noise_image_path = f"E://collage_project//project2//images//{img_name}_noisy_image.png"
noise_image = Image.open(noise_image_path)
noise_image = noise_image.convert('L')
noise_np_image = np.array(noise_image,dtype=np.uint8)

dec_img = decrypt_image(noise_np_image,dec_key)
decrypted_image_path = f"E://collage_project//project2//images//{img_name}_decrypted_noisy_image.png"
Image.fromarray(dec_img).save(decrypted_image_path)
print(f"Image saved at: {decrypted_image_path}")