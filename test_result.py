import math
import cv2
from collections import Counter
from skimage.metrics import peak_signal_noise_ratio
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy.stats import chisquare

# Histogram: Generate and save a comparative visualization of original and encrypted images 
# with their respective pixel intensity histograms.
def histograms(original_image, encrypted_image):
    save_path = f"E://collage_project//project2//images//{img_name}_combined_histogram.png"

    # Calculate histograms for original and encrypted images
    hist_original = np.histogram(original_image.ravel(), bins=256, range=(0, 256))[0]
    hist_encrypted = np.histogram(encrypted_image.ravel(), bins=256, range=(0, 256))[0]

    # Create a visualization with images and their histograms
    plt.figure(figsize=(24, 6))

    # Plot original image
    plt.subplot(1, 4, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title("Original Image", fontsize=16)
    plt.axis('off')

    # Plot original image histogram
    plt.subplot(1, 4, 2)
    plt.bar(range(256), hist_original, color='blue', alpha=0.7)
    plt.title("Histogram of Original Image", fontsize=16)
    plt.xlabel("Pixel Intensity", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xlim(0, 255)
    plt.ylim(0, max(hist_original.max(), hist_encrypted.max()))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Plot encrypted image
    plt.subplot(1, 4, 3)
    plt.imshow(encrypted_image, cmap='gray')
    plt.title("Encrypted Image", fontsize=16)
    plt.axis('off')

    # Plot encrypted image histogram
    plt.subplot(1, 4, 4)
    plt.bar(range(256), hist_encrypted, color='red', alpha=0.7)
    plt.title("Histogram of Encrypted Image", fontsize=16)
    plt.xlabel("Pixel Intensity", fontsize=14)
    plt.ylabel("Frequency", fontsize=14)
    plt.xlim(0, 255)
    plt.ylim(0, max(hist_original.max(), hist_encrypted.max()))
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)

    # Save and display the visualization
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    print(f"Combined histogram saved at: {save_path}")
    plt.show()

# Calculate pixel correlation in different directions (horizontal, vertical, diagonal).
def correlation_coefficient(image):
    M, N = image.shape[:2]
    # Horizontal correlation: between adjacent pixels in rows
    x_horizontal = image[:, :-1].flatten()
    y_horizontal = image[:, 1:].flatten()
    corr_horizontal = np.corrcoef(x_horizontal, y_horizontal)[0, 1]
    # Vertical correlation: between adjacent pixels in columns
    x_vertical = image[:-1, :].flatten()
    y_vertical = image[1:, :].flatten()
    corr_vertical = np.corrcoef(x_vertical, y_vertical)[0, 1]
    # Diagonal correlation: between adjacent diagonal pixels
    x_diagonal = image[:-1, :-1].flatten()
    y_diagonal = image[1:, 1:].flatten()
    corr_diagonal = np.corrcoef(x_diagonal, y_diagonal)[0, 1]
    # Return average correlation coefficient
    return corr_horizontal , corr_vertical , corr_diagonal

def calculate_mse(original_image, encrypted_image):
    # Ensure both images are of the same size
    assert original_image.shape == encrypted_image.shape, "Images must have the same dimensions"
    
    # Calculate the Mean Squared Error
    mse = np.mean((original_image - encrypted_image) ** 2)
    return mse

# Calculate the entropy of an image to measure its randomness.
def calculate_entropy(image_matrix):
    # Flatten image and count pixel value occurrences
    flattened_image = image_matrix.flatten()
    pixel_counts = Counter(flattened_image)
    total_pixels = len(flattened_image)

    # Compute entropy based on pixel probability
    entropy = 0.0
    for count in pixel_counts.values():
        probability = count / total_pixels
        entropy -= probability * math.log2(probability)

    return entropy

# Calculate Number of Pixels Change Rate to measure encryption sensitivity.
def calculate_npcr(original_image, encrypted_image):
    # Ensure images have same dimensions
    if original_image.shape != encrypted_image.shape:
        raise ValueError("Images must have same dimensions.")

    # Calculate percentage of different pixels
    diff_pixels = np.sum(original_image != encrypted_image)
    total_pixels = original_image.size
    npcr = (diff_pixels / total_pixels) * 100

    return npcr

# Calculate Unified Average Changing Intensity to measure encryption diversity.
def calculate_uaci(image1, image2):
    assert image1.shape == image2.shape, "Images must have same dimensions"
    
    # Calculate absolute pixel difference and percentage change
    M, N = image1.shape
    diff = np.abs(image1.astype(np.int16) - image2.astype(np.int16))
    uaci = (np.sum(diff) / (M * N * 255)) * 100
    
    return uaci

def chi_square_test(image):
    
    # Flatten to 1D array
    pixel_values = image.flatten()
    
    # Count occurrences of each intensity level (0–255)
    observed_freq = np.bincount(pixel_values, minlength=256)
    
    # Expected frequency assuming uniform distribution
    expected_freq = np.full(256, fill_value=len(pixel_values)/256)
    
    # Perform Chi-Square test
    chi_stat, p_value = chisquare(f_obs=observed_freq, f_exp=expected_freq)
    
    return chi_stat



def calculate_test(gray_image, encrypted_image, encrypted_image2, decrypted_image):

    # Generate and save histogram visualization
    histograms(gray_image, encrypted_image)
    
    # Analyze pixel correlations in encrypted image
    corr_horizontal, corr_vertical, corr_diagonal = correlation_coefficient(encrypted_image)
    print(f"Correlation Coefficient Horizontal: {corr_horizontal}")
    print(f"Correlation Coefficient Vertical: {corr_vertical}")
    print(f"Correlation Coefficient Diagonal: {corr_diagonal}")
    
    # Calculate encryption quality metrics
    entropy = calculate_entropy(encrypted_image)
    print("Entropy: ", entropy)
    
    mse = calculate_mse(gray_image, encrypted_image)
    print(f"MSE: {mse}")

    npcr_value = calculate_npcr(encrypted_image, encrypted_image2)
    print(f"NPCR value: {npcr_value:.4f}%")
    
    uaci_value = calculate_uaci(encrypted_image, encrypted_image2)
    print(f"UACI value: {uaci_value:.4f}%")

    # Calculate image quality metrics
    psnr_original_encrypted = peak_signal_noise_ratio(gray_image, encrypted_image)
    print(f"PSNR (original vs encrypted): {psnr_original_encrypted}")
    
    psnr_original_decrypted = peak_signal_noise_ratio(gray_image, decrypted_image)
    print(f"PSNR (original vs decrypted): {psnr_original_decrypted}")
    
    ssim_value = ssim(gray_image, decrypted_image)
    print(f"SSIM: {ssim_value}")
    
    chi_value_gray = chi_square_test(gray_image)
    print("Chi square value for gray image: ",chi_value_gray)
    
    chi_value_encrypted = chi_square_test(encrypted_image)
    print("Chi square value for encrypted image: ",chi_value_encrypted)

# Main script for loading and analyzing images
img_name = "pepper"
path = f"E://collage_project//project2//images//{img_name}_gray_image.png"
gray_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)

path = f"E://collage_project//project2//images//{img_name}_encrypted_hfpso.png"
encrypted_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

path = f"E://collage_project//project2//images//modified_{img_name}_encrypted_hfpso.png"
encrypted_image2 = cv2.imread(path, cv2.IMREAD_GRAYSCALE) 

path = f"E://collage_project//project2//images//{img_name}_decrypted_hfpso.png"
decrypted_image = cv2.imread(path, cv2.IMREAD_GRAYSCALE)  

# Perform comprehensive image encryption analysis
calculate_test(gray_image, encrypted_image, encrypted_image2, decrypted_image)