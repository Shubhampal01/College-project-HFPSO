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
## 📤 Expected Output

After running the `hybrid_HFPSO.py` script, the following outputs will be generated:

### 🖼️ Output Images

- **Encrypted Image**  
  Saved at:```bash
images/{img_name}_encrypted_hfpso.png```
- **Decrypted Image**  
Saved at:```bash
images/{img_name}_decrypted_hfpso.png```
> Replace `{img_name}` with the actual image name you specified (e.g., `lena`, `pepper`).

---

### 🧾 Console Output

The terminal/console will display a series of outputs including:

- **HFPSO Optimization Progress**
- Iteration-wise fitness values
- Key generation and convergence status

- **Image Quality & Security Metrics**
- **Correlation Coefficients** (between adjacent pixels)
- **PSNR** (Peak Signal-to-Noise Ratio)
- **MSE** (Mean Squared Error)
- **Entropy** (information randomness)
- **UACI** (Unified Average Changing Intensity)
- **NPCR** (Number of Pixels Change Rate), if implemented

- **Decryption Status Message**
