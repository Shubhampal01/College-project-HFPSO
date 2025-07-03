import numpy as np
import random
from PIL import Image
import math
from skimage.metrics import peak_signal_noise_ratio as psnr

# Key generation: Uses a block of 5 consecutive pixels from the image to calculate an initial key.
def generate_key(np_image, num):
    M, N = np_image.shape[:2]
    # Calculate the starting pixel for the 5-pixel block
    row = num // N
    col = num % N
    
    # Take a block of 5 consecutive pixels
    block = []
    for i in range(5):
        if col + i < N:
            block.append(np_image[row, col + i])
        else:
            # Wrap around to the next row if out of bounds
            block.append(np_image[(row + 1) % M, (col + i) % N])
    
    # Calculate the key by sum of the 5 pixels
    key = sum(block) / (256 * len(block))
    return key

# Logistic map key sequence generation: Generates a sequence of key values using the logistic map
def logistic_map_key_sequence(size, r, x0):
    key_sequence = np.zeros(size, dtype=np.uint8)
    x = x0
    # Iterate a large number of times to ensure the system is chaotic
    for i in range(10000):
        x = r * x * (1 - x) 
    # Generate the actual key sequence using the logistic map formula
    for i in range(size):
        x = r * x * (1 - x)  # Logistic map iteration
        key_sequence[i] = int(x * 256) % 256  
    return key_sequence

# Forward diffusion: Applies a forward diffusion process on the image using a key sequence
def forward_diffusion_gray(image, key_sequence):
    image_1d = image.reshape(-1)
    image_F = image_1d.copy()
    # Apply forward diffusion starting with the first pixel
    image_F[0] = 0 ^ key_sequence[0] ^ image_1d[0]
    # Applying the diffusion operation on subsequent pixels
    for i in range(1, len(image_1d)):
        image_F[i] = image_F[i - 1] ^ key_sequence[i] ^ image_1d[i]
    return image_F.reshape(image.shape)

# Backward diffusion: Applies the backward diffusion process on the image using the key sequence
def backward_diffusion_gray(image_F, key_sequence):
    image_F_1d = image_F.reshape(-1)
    image_B = image_F_1d.copy()
    # Apply backward diffusion, starting with the last pixel
    lastInd = len(image_F_1d) - 1
    image_B[lastInd] = 0 ^ key_sequence[lastInd] ^ image_F_1d[lastInd]
    # Applying backward diffusion on previous pixels
    for i in range(len(image_F_1d) - 2, -1, -1):
        image_B[i] = image_B[i + 1] ^ key_sequence[i] ^ image_F_1d[i]
    return image_B.reshape(image_F.shape)

# Reverse backward diffusion: Reverses the backward diffusion process on the image
def reverse_backward_diffusion_gray(image_F, key_sequence):
    image_F_1d = image_F.reshape(-1)
    image_B = image_F_1d.copy()
    # Reverse backward diffusion, starting with the last pixel
    lastInd = len(image_F_1d) - 1
    image_B[lastInd] = 0 ^ key_sequence[lastInd] ^ image_F_1d[lastInd]
    # reversing the backward diffusion on previous pixels
    for i in range(len(image_F_1d) - 2, -1, -1):
        image_B[i] = image_F_1d[i + 1] ^ key_sequence[i] ^ image_F_1d[i]
    return image_B.reshape(image_F.shape)

# Reverse forward diffusion: Reverses the forward diffusion process on the image
def reverse_forward_diffusion_gray(image, key_sequence):
    image_1d = image.reshape(-1)
    image_F = image_1d.copy()
    # Reverse forward diffusion starting with the first pixel
    image_F[0] = 0 ^ key_sequence[0] ^ image_1d[0]
    # reversing forward diffusion on subsequent pixels
    for i in range(1, len(image_1d)):
        image_F[i] = image_1d[i] ^ key_sequence[i] ^ image_1d[i - 1]
    return image_F.reshape(image.shape)

# Encrypt image using a logistic map-based key sequence and diffusion processes
def encrypt_image(np_image, x0, r=3.8):
    size = np_image.size
    # Generate a key sequence using the logistic map
    key_sequence = logistic_map_key_sequence(size, r, x0)
    
    # Apply forward and backward diffusion to the image
    forward_diffused_image = forward_diffusion_gray(np_image, key_sequence)
    backward_diffused_image = backward_diffusion_gray(forward_diffused_image, key_sequence)
    # Shuffle the pixels
    en_shuffle_img = shuffle_pixels(backward_diffused_image, x0=x0, r=r)
    
    return en_shuffle_img

# Decrypt the encrypted image using the reverse diffusion process and key sequence
def decrypt_image(np_image, x0, r=3.8):
    size = np_image.size
    key_sequence = logistic_map_key_sequence(size, r, x0)

    # Unshuffle the pixels to reverse the encryption process
    dec_shuffle_img = unshuffle_pixels(np_image, x0=x0, r=r)
    rev_back_diffused_image = reverse_backward_diffusion_gray(dec_shuffle_img, key_sequence)
    rev_ford_diffused_image = reverse_forward_diffusion_gray(rev_back_diffused_image, key_sequence)
    
    return rev_ford_diffused_image

# Compute the correlation coefficient between adjacent pixels to measure image randomness
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
    return (corr_horizontal + corr_vertical + corr_diagonal)/3

# Generate a logistic map-based sequence of indices for pixel shuffling
def logistic_map_sequence(length, x0=0.5, r=3.99):
    sequence = []
    x = x0
    # Iterate for a large number of steps to ensure chaotic behavior
    for i in range(10000):
        x = r * x * (1 - x) 
    # Generate the sequence of indices by applying the logistic map
    for _ in range(length):
        x = r * x * (1 - x)
        sequence.append(x)
    # Normalize the values to indices in the range of the image's size
    indices = np.argsort(sequence)
    return indices

# Shuffle pixels based on the logistic map sequence
def shuffle_pixels(en_img, x0=0.5, r=3.99):
    flat_image = en_img.flatten()
    total_pixels = flat_image.size
    
    # Generate the shuffle sequence using the logistic map
    shuffle_indices = logistic_map_sequence(total_pixels, x0=x0, r=r)
    
    # Apply the shuffle by reordering the pixels
    shuffled_flat_image = flat_image[shuffle_indices]
    
    # Reshape back to the original image dimensions
    shuffled_image = shuffled_flat_image.reshape(en_img.shape)
    return shuffled_image

# Unshuffle the pixels to reverse the shuffle operation
def unshuffle_pixels(shuffled_image, x0=0.5, r=3.99):
    flat_image = shuffled_image.flatten()
    total_pixels = flat_image.size
    
    # Generate the logistic map sequence for unshuffling
    shuffle_indices = logistic_map_sequence(total_pixels, x0=x0, r=r)
    
    # Reverse the shuffle by placing pixels back to their original positions
    unshuffled_flat_image = np.zeros_like(flat_image)
    unshuffled_flat_image[shuffle_indices] = flat_image
    
    # Reshape back to the original image dimensions
    unshuffled_image = unshuffled_flat_image.reshape(shuffled_image.shape)
    return unshuffled_image

# Hybrid Firefly and Particle Swarm Optimization (HFPSO) Algorithm
def HFPSO(swarm_size, max_iterations, c1=1.49445, c2=1.49445, vmax_coef=0.1):
    """
    Hybrid Firefly and Particle Swarm Optimization for finding optimal encryption key position
    """
    # Initialize bounds for pixel positions
    LB = 0
    UB = total_pixels - 1
    
    # Initialize velocity limits
    v_max = vmax_coef * (UB - LB)
    v_min = -v_max
    
    # Initialize particle positions and velocities
    particles_x = np.random.randint(LB, UB + 1, size=swarm_size)
    particles_v = np.random.uniform(v_min, v_max, size=swarm_size)
    
    # Evaluate initial fitness
    f_val = np.zeros(swarm_size)
    for i in range(swarm_size):
        key = generate_key(np_image, particles_x[i])
        encrypted_image = encrypt_image(np_image, key)
        f_val[i] = abs(correlation_coefficient(encrypted_image))
    
    # Initialize personal and global bests
    p_best = particles_x.copy()
    p_best_val = f_val.copy()
    
    # Find global best
    best_index = np.argmin(f_val)
    g_best = particles_x[best_index]
    g_best_val = f_val[best_index]
    
    # Calculate maximum distance for firefly algorithm
    dmax = (UB - LB)
    
    # Store history for firefly behavior
    g_best_history = []
    g_best_val_history = []
    
    print("Starting HFPSO Optimization...")
    
    for iteration in range(max_iterations):
        # Linear decreasing inertia weight
        w = 0.9 - ((0.9 - 0.5) / max_iterations) * iteration
        
        for j in range(swarm_size):
            # Check if particle should use firefly behavior (after 2 iterations and no improvement)
            use_firefly = False
            if iteration > 2 and len(g_best_val_history) >= 3:
                if f_val[j] <= g_best_val_history[iteration - 2]:
                    use_firefly = True
            
            if use_firefly and len(g_best_history) >= 3:
                # Firefly Algorithm behavior
                rij = abs(particles_x[j] - g_best_history[iteration - 2]) / dmax
                
                # Firefly parameters
                alpha = 0.2
                beta0 = 2
                m = 2
                gamma = 1
                
                # Attractiveness calculation
                beta = beta0 * np.exp(-gamma * (rij ** m))
                
                # Random factor
                e = np.random.random() - 0.5
                
                # Update position using firefly movement
                prev_pos = particles_x[j]
                particles_x[j] = particles_x[j] + beta * (particles_x[j] - g_best_history[iteration - 2]) + alpha * e
                
                # Boundary constraint
                particles_x[j] = max(LB, min(UB, int(particles_x[j])))
                
                # Update velocity based on position change
                particles_v[j] = particles_x[j] - prev_pos
                
            else:
                # Standard PSO behavior
                r1 = np.random.random()
                r2 = np.random.random()
                
                # Update velocity
                particles_v[j] = (w * particles_v[j] + 
                                c1 * r1 * (p_best[j] - particles_x[j]) + 
                                c2 * r2 * (g_best - particles_x[j]))
                
                # Update position
                particles_x[j] = particles_x[j] + particles_v[j]
            
            # Apply velocity constraints
            particles_v[j] = max(v_min, min(v_max, particles_v[j]))
            
            # Apply position constraints
            particles_x[j] = max(LB, min(UB, int(particles_x[j])))
        
        # Evaluate fitness for all particles
        for i in range(swarm_size):
            key = generate_key(np_image, particles_x[i])
            encrypted_image = encrypt_image(np_image, key)
            f_val[i] = abs(correlation_coefficient(encrypted_image))
        
        # Update personal and global bests
        for j in range(swarm_size):
            if f_val[j] < p_best_val[j]:
                p_best[j] = particles_x[j]
                p_best_val[j] = f_val[j]
            
            if p_best_val[j] < g_best_val:
                g_best = particles_x[j]
                g_best_val = p_best_val[j]
        
        # Store history
        g_best_history.append(g_best)
        g_best_val_history.append(g_best_val)
        
        print(f"Iteration {iteration + 1}: Best Fitness = {g_best_val}, Best Position = {g_best}")
    
    # Generate final encrypted image with best key
    best_key = generate_key(np_image, g_best)
    best_encrypted_image = encrypt_image(np_image, best_key)
    
    return best_encrypted_image, g_best, g_best_val

#     """
#     Hybrid Firefly and Particle Swarm Optimization for finding optimal encryption key position
#     """
#     # Initialize bounds for pixel positions
#     LB = 0
#     UB = total_pixels - 1
    
#     # Initialize velocity limits
#     v_max = vmax_coef * (UB - LB)
#     v_min = -v_max
    
#     # Initialize particle positions and velocities
#     particles_x = np.random.randint(LB, UB + 1, size=swarm_size)
#     particles_v = np.random.uniform(v_min, v_max, size=swarm_size)
    
#     # Evaluate initial fitness (convert to maximization problem for firefly)
#     f_val = np.zeros(swarm_size)
#     for i in range(swarm_size):
#         key = generate_key(np_image, particles_x[i])
#         encrypted_image = encrypt_image(np_image, key)
#         corr_coef = abs(correlation_coefficient(encrypted_image))
#         f_val[i] = corr_coef  # Keep as minimization for PSO
    
#     # Initialize personal and global bests
#     p_best = particles_x.copy()
#     p_best_val = f_val.copy()
    
#     # Find global best (minimum correlation)
#     best_index = np.argmin(f_val)
#     g_best = particles_x[best_index]
#     g_best_val = f_val[best_index]
    
#     # Calculate maximum distance for firefly algorithm
#     dmax = (UB - LB)
    
#     # Store history for convergence tracking
#     g_best_history = [g_best]
#     g_best_val_history = [g_best_val]
    
#     # Track stagnation for firefly switching
#     stagnation_counter = np.zeros(swarm_size)
#     stagnation_threshold = 3
    
#     print("Starting HFPSO Optimization...")
#     print(f"Initial Best Fitness = {g_best_val}, Best Position = {g_best}")
    
#     for iteration in range(max_iterations):
#         # Linear decreasing inertia weight
#         w = 0.9 - ((0.9 - 0.4) / max_iterations) * iteration
        
#         for j in range(swarm_size):
#             # Determine if particle should use firefly behavior
#             # Use firefly if particle has stagnated (no improvement for several iterations)
#             use_firefly = stagnation_counter[j] >= stagnation_threshold
            
#             if use_firefly:
#                 # Reset stagnation counter
#                 stagnation_counter[j] = 0
                
#                 # Firefly Algorithm behavior
#                 # Find the best particle (brightest firefly) to move towards
#                 best_particle_idx = np.argmin(f_val)
                
#                 if j != best_particle_idx:  # Don't move if already the best
#                     # Calculate distance
#                     distance = abs(particles_x[j] - particles_x[best_particle_idx])
#                     rij = distance / dmax if dmax > 0 else 0
                    
#                     # Firefly parameters
#                     alpha = 0.2  # Random movement factor
#                     beta0 = 2    # Maximum attractiveness
#                     gamma = 1    # Light absorption coefficient
                    
#                     # Attractiveness calculation (decreases with distance)
#                     beta = beta0 * np.exp(-gamma * (rij ** 2))
                    
#                     # Random factor
#                     epsilon = np.random.uniform(-0.5, 0.5)
                    
#                     # Move towards brighter firefly (better solution)
#                     direction = 1 if particles_x[best_particle_idx] > particles_x[j] else -1
#                     new_position = (particles_x[j] + 
#                                   beta * direction * distance + 
#                                   alpha * epsilon * (UB - LB))
                    
#                     # Update position
#                     particles_x[j] = int(np.clip(new_position, LB, UB))
                    
#                     # Update velocity based on position change
#                     particles_v[j] = new_position - particles_x[j]
                
#             else:
#                 # Standard PSO behavior
#                 r1 = np.random.random()
#                 r2 = np.random.random()
                
#                 # Update velocity
#                 particles_v[j] = (w * particles_v[j] + 
#                                 c1 * r1 * (p_best[j] - particles_x[j]) + 
#                                 c2 * r2 * (g_best - particles_x[j]))
                
#                 # Apply velocity constraints
#                 particles_v[j] = np.clip(particles_v[j], v_min, v_max)
                
#                 # Update position
#                 new_position = particles_x[j] + particles_v[j]
#                 particles_x[j] = int(np.clip(new_position, LB, UB))
        
#         # Evaluate fitness for all particles
#         prev_f_val = f_val.copy()
#         for i in range(swarm_size):
#             key = generate_key(np_image, particles_x[i])
#             encrypted_image = encrypt_image(np_image, key)
#             f_val[i] = abs(correlation_coefficient(encrypted_image))
        
#         # Update personal bests and stagnation counters
#         for j in range(swarm_size):
#             if f_val[j] < p_best_val[j]:
#                 p_best[j] = particles_x[j]
#                 p_best_val[j] = f_val[j]
#                 stagnation_counter[j] = 0  # Reset stagnation
#             else:
#                 stagnation_counter[j] += 1  # Increment stagnation
        
#         # Update global best
#         current_best_idx = np.argmin(p_best_val)
#         if p_best_val[current_best_idx] < g_best_val:
#             g_best = p_best[current_best_idx]
#             g_best_val = p_best_val[current_best_idx]
        
#         # Store history
#         g_best_history.append(g_best)
#         g_best_val_history.append(g_best_val)
        
#         print(f"Iteration {iteration + 1}: Best Fitness = {g_best_val:.6f}, Best Position = {g_best}")
        
#         # Early stopping if very good solution found
#         if g_best_val < 1e-6:
#             print(f"Excellent solution found at iteration {iteration + 1}")
#             break
    
#     # Generate final encrypted image with best key
#     best_key = generate_key(np_image, g_best)
#     best_encrypted_image = encrypt_image(np_image, best_key)
    
#     print(f"\nOptimization completed!")
#     print(f"Final Best Position: {g_best}")
#     print(f"Final Best Fitness: {g_best_val:.6f}")
    
#     return best_encrypted_image, g_best, g_best_val

# Example usage function with required dependencies
def example_usage():
    """
    Example of how to use the corrected HFPSO function
    You need to implement these functions based on your specific encryption scheme
    """
    
    def generate_key(image, position):
        """Generate encryption key based on pixel position"""
        # Implement your key generation logic here
        # This is just a placeholder
        return np.random.randint(0, 256, size=image.shape)
    
    def encrypt_image(image, key):
        """Encrypt image using the generated key"""
        # Implement your encryption logic here
        # This is just a placeholder
        return np.bitwise_xor(image, key)
    
    def correlation_coefficient(image):
        """Calculate correlation coefficient of encrypted image"""
        # Implement correlation calculation
        # This is just a placeholder
        flat_image = image.flatten()
        if len(flat_image) < 2:
            return 0
        return np.corrcoef(flat_image[:-1], flat_image[1:])[0, 1]
    
    # Example parameters
    swarm_size = 30
    max_iterations = 100
    total_pixels = 256 * 256  # For a 256x256 image
    np_image = np.random.randint(0, 256, (256, 256), dtype=np.uint8)
    
    # Run optimization
    encrypted_img, best_pos, best_fitness, history = HFPSO(
        swarm_size, max_iterations, total_pixels, np_image,
        generate_key, encrypt_image, correlation_coefficient
    )
    
    return encrypted_img, best_pos, best_fitness, history
# Entropy calculation
def calculate_entropy(image):
    histogram = np.bincount(image.flatten(), minlength=256)
    total_pixels = image.size
    entropy = 0
    for count in histogram:
        if count == 0:
            continue
        p = count / total_pixels
        entropy -= p * math.log2(p)
    return entropy

# UACI (Unified Average Changing Intensity)
def calculate_uaci(original, encrypted):
    diff = np.abs(original.astype(int) - encrypted.astype(int))
    uaci = np.sum(diff) / (original.size * 255)
    return uaci * 100  # Convert to percentage


# Main function to demonstrate encryption and decryption
if __name__ == "__main__":
    img_name = "lena"
    img_path = f"E://collage_project//project2//images//{img_name}.png"
    image = Image.open(img_path)
    image = image.convert('L')  # Convert to grayscale
    np_image = np.array(image, dtype=np.uint8)
    M, N = np_image.shape[0], np_image.shape[1]
    total_pixels = M * N
    
    print(f"Image dimensions: {M}x{N}")
    print(f"Total pixels: {total_pixels}")
    
    # HFPSO parameters
    swarm_size = 10
    max_iterations = 10
    
    # Run HFPSO optimization to find the best encryption key
    en_img, best_position, best_fitness = HFPSO(swarm_size, max_iterations)
    
    print(f"\nOptimization completed!")
    print(f"Best Position: {best_position}")
    print(f"Best Fitness (Correlation): {best_fitness:.6f}")
    
    # Save the encrypted image
    def save_image(image, desc):
        image_path = f"E://collage_project//project2//images//{img_name}_{desc}.png"
        Image.fromarray(image).save(image_path)
        print(f"Image saved at: {image_path}")

    save_image(en_img, "encrypted_hfpso")

    # Decrypt the image using the best position found during optimization
    dec_key = generate_key(np_image, int(best_position))
    decrypted_image = decrypt_image(en_img, dec_key)
    save_image(decrypted_image, "decrypted_hfpso")
    
    # Calculate correlation coefficient for original and encrypted images
    original_corr = correlation_coefficient(np_image)
    encrypted_corr = correlation_coefficient(en_img)
    decrypted_corr = correlation_coefficient(decrypted_image)
    
    print(f"\n--- Performance Metrics ---")
    print(f"\nCorrelation Analysis:")
    print(f"Original Image Correlation: {original_corr}")
    print(f"Encrypted Image Correlation: {encrypted_corr}")
    print(f"Decrypted Image Correlation: {decrypted_corr}")
    # PSNR between original and decrypted
    psnr_val = psnr(np_image, en_img, data_range=255)
    print(f"PSNR (Original vs Encrypted): {psnr_val:.2f} dB")
    
    # Entropy of original and encrypted images
    entropy_original = calculate_entropy(np_image)
    entropy_encrypted = calculate_entropy(en_img)
    print(f"Entropy - Original: {entropy_original:.4f}")
    print(f"Entropy - Encrypted: {entropy_encrypted:.4f}")
    
    # UACI between original and encrypted
    uaci_val = calculate_uaci(np_image, en_img)
    print(f"UACI (Original vs Encrypted): {uaci_val:.2f} %")
    
    # Verify decryption accuracy
    mse = np.mean((np_image.astype(float) - decrypted_image.astype(float)) ** 2)
    print(f"Mean Squared Error (Original vs Decrypted): {mse}")
    
    if mse < 1e-10:
        print("✓ Perfect decryption achieved!")
    else:
        print("⚠ Decryption has some errors")