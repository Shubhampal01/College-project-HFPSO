# Image Encryption and Decryption using Hybrid nature inspired optimization Project

This project implements an image encryption and decryption scheme using a hybrid Firefly and Particle Swarm Optimization (HFPSO) algorithm to find optimal encryption keys. It also includes scripts for analyzing the security of the encryption against common attacks like cropping and Gaussian noise, and for evaluating various image quality and security metrics.
### Prerequisites

Before running the project, ensure you have the following installed:

*   **Python 3.x**
*   **Required Python Libraries**:
    *   `numpy`
    *   `Pillow` (PIL)
    *   `matplotlib`
    *   `scikit-image` (`skimage`)
    *   `scipy`
    *   `opencv-python` (`cv2`)

You can install these libraries using pip:
```bash
pip install numpy Pillow matplotlib scikit-image scipy opencv-python
```

## 🚀 Running the Project

### 🔐 Encrypt and Decrypt an Image

This is the core functionality, using the `hybrid_HFPSO.py` script to apply encryption and then decrypt the image.

#### 📋 Instructions

1. Open the `hybrid_HFPSO.py` script in your preferred code editor.
2. Modify the `img_name` variable to point to your input image (without file extension), for example:

```python
img_name = "lena"
```
3. Run the script from your terminal:

```bash
python hybrid_HFPSO.py
```
## 📤 Expected Output

After running the `hybrid_HFPSO.py` script, the following outputs will be generated:
- **Encrypted Image**  
  Saved at:```
images/{img_name}_encrypted_hfpso.png```
- **Decrypted Image**  
Saved at:```
images/{img_name}_decrypted_hfpso.png```
> Replace `{img_name}` with the actual image name you specified (e.g., `lena`, `pepper`).

---
## 📊 Analyze Encryption Performance

The `test_result.py` script calculates various **security** and **image quality** metrics to evaluate the performance of the encryption scheme.

---

### 🧪 How to Run

1. Open the `test_result.py` script in your code editor.
2. Set the `img_name` variable to the name of the image you want to analyze (without file extension). For example:

```python
# In test_result.py
img_name = "pepper"
```
3.Execute the script using the terminal:

```bash
python test_result.py
```
## Expected Output

### 📊 Combined Histogram Plot
- A combined histogram plot showing the **original** and **encrypted** image histograms.
- Saved as:
```images/{img_name}_combined_histogram.png```
---

### 🖥️ Console Output

#### 🔗 Correlation Coefficients

#### 📈 Entropy  -Displays the entropy of the image to measure randomness

#### 🧮 Mean Squared Error (MSE)   -Measures the average of the squares of the differences between original and processed images

#### 🔄 Number of Pixels Change Rate (NPCR)   - Indicates how many pixel values have changed between original and encrypted images

#### 📊 Unified Average Changing Intensity (UACI)  - Measures the average intensity difference between the original and encrypted images

#### 🔉 Peak Signal-to-Noise Ratio (PSNR)

#### 🧠 Structural Similarity Index (SSIM)  - Quantifies the similarity between original and processed images.
