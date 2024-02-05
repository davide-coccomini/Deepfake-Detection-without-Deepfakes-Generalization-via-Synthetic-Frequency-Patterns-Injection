import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transform.albu import FrequencyPatterns

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

image_path = 'fingerprints/patterns/dog.png'
base_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

output_folder = 'fingerprints/new_patterns'
os.makedirs(output_folder, exist_ok=True)

base_image_gray = rgb2gray(base_image)
base_image_fft = np.fft.fftshift(np.fft.fft2(base_image_gray))

base_image_fft_normalized = np.log(np.abs(base_image_fft) + 1)
base_image_fft_normalized = ((base_image_fft_normalized - base_image_fft_normalized.min()) /
                              (base_image_fft_normalized.max() - base_image_fft_normalized.min()) * 255).astype(np.uint8)

output_path_base_fft = os.path.join(output_folder, 'base_image_fft.png')
plt.imsave(output_path_base_fft, base_image_fft_normalized, cmap='viridis')
print(f'FFT dell\'immagine di base salvata in: {output_path_base_fft}')

freq_patterns = FrequencyPatterns(p=1.0)  
already_seen = []
samples = 1
for i, pattern_func in enumerate(freq_patterns.patterns):
    for j in range(samples):  
        if (pattern_func.__name__ in already_seen and samples == 1):
            continue
        else:
            already_seen.append(pattern_func.__name__)
        pattern, make_pattern_fft = freq_patterns.apply(img=base_image, required_pattern_fn=pattern_func, return_pattern=True, mode=0)
        #pattern = pattern.astype(np.uint8)
        #pattern_rgb = cv2.cvtColor(pattern, cv2.COLOR_GRAY2RGB)
        if not make_pattern_fft: # The pattern is already FFT so I make the IFFT
            pattern = np.fft.ifft2(pattern).real


        print("Applying", pattern_func.__name__)
        output_pattern = os.path.join(output_folder, f'{pattern_func.__name__}_pattern_{j + 1}.png')
        plt.imsave(output_pattern, pattern)
        print(f'Pattern {pattern_func.__name__} applicato salvato in: {output_pattern}')


        f_pattern = np.fft.fftshift(np.fft.fft2(pattern))
        f_pattern_normalized = np.log(np.abs(f_pattern) + 1)
        f_pattern_normalized = ((f_pattern_normalized - f_pattern_normalized.min()) /
                                           (f_pattern_normalized.max() - f_pattern_normalized.min()) * 255).astype(np.uint8)


        output_f_pattern = os.path.join(output_folder, f'{pattern_func.__name__}_pattern_fft_{j + 1}.png')
        plt.imsave(output_f_pattern, f_pattern_normalized.real, cmap='viridis')
        print(f'FFT Pattern {pattern_func.__name__} applicato salvato in: {output_f_pattern}')

        image_with_pattern = freq_patterns.apply(img=base_image, required_pattern_fn=pattern_func, return_pattern=False, mode=0)

        image_with_pattern = image_with_pattern.astype(np.uint8)

        image_with_pattern_rgb = cv2.cvtColor(image_with_pattern, cv2.COLOR_BGR2RGB)
            
        output_path_image_with_pattern = os.path.join(output_folder, f'{pattern_func.__name__}_image_with_pattern_{j + 1}.png')
        plt.imsave(output_path_image_with_pattern, image_with_pattern_rgb)
        print(f'Immagine con il Pattern {pattern_func.__name__} applicato salvata in: {output_path_image_with_pattern}')

        gray_image_with_pattern = rgb2gray(image_with_pattern)

        image_with_pattern_fft = np.fft.fftshift(np.fft.fft2(gray_image_with_pattern))

        image_with_pattern_fft_normalized = np.log(np.abs(image_with_pattern_fft) + 1)
        image_with_pattern_fft_normalized = ((image_with_pattern_fft_normalized - image_with_pattern_fft_normalized.min()) /
                                           (image_with_pattern_fft_normalized.max() - image_with_pattern_fft_normalized.min()) * 255).astype(np.uint8)

        output_path_image_with_pattern_fft = os.path.join(output_folder, f'{pattern_func.__name__}_image_with_pattern_fft_{j + 1}.png')
        plt.imsave(output_path_image_with_pattern_fft, image_with_pattern_fft_normalized.real, cmap='viridis')
        print(f'FFT dell\'immagine con il Pattern {pattern_func.__name__} applicato salvata in: {output_path_image_with_pattern_fft}')


        '''
        image_with_pattern = freq_patterns.apply(img=base_image, required_pattern_fn=pattern_func, return_pattern=False, mode=1)

        image_with_pattern = image_with_pattern.astype(np.uint8)

        image_with_pattern_rgb = cv2.cvtColor(image_with_pattern, cv2.COLOR_BGR2RGB)

        output_path_image_with_pattern = os.path.join(output_folder, f'{pattern_func.__name__}_image_with_pattern_{j + 1}_mode1.png')
        plt.imsave(output_path_image_with_pattern, image_with_pattern_rgb)
        print(f'Immagine con il Pattern {pattern_func.__name__} applicato salvata in: {output_path_image_with_pattern}')

        gray_image_with_pattern = rgb2gray(image_with_pattern)

        image_with_pattern_fft = np.fft.fftshift(np.fft.fft2(gray_image_with_pattern))

        image_with_pattern_fft_normalized = np.log(np.abs(image_with_pattern_fft) + 1)
        #image_with_pattern_fft_normalized = ((image_with_pattern_fft_normalized - image_with_pattern_fft_normalized.min()) /
        #                                   (image_with_pattern_fft_normalized.max() - image_with_pattern_fft_normalized.min()) * 255).astype(np.uint8)

        output_path_image_with_pattern_fft = os.path.join(output_folder, f'{pattern_func.__name__}_image_with_pattern_fft_{j + 1}_mode1.png')
        plt.imsave(output_path_image_with_pattern_fft, image_with_pattern_fft_normalized, cmap='viridis')
        print(f'FFT dell\'immagine con il Pattern {pattern_func.__name__} applicato salvata in: {output_path_image_with_pattern_fft}')




        image_with_pattern = freq_patterns.apply(img=base_image, required_pattern_fn=pattern_func, return_pattern=False, mode=2)

        image_with_pattern = image_with_pattern.astype(np.uint8)

        image_with_pattern_rgb = cv2.cvtColor(image_with_pattern, cv2.COLOR_BGR2RGB)

        output_path_image_with_pattern = os.path.join(output_folder, f'{pattern_func.__name__}_image_with_pattern_{j + 1}_mode2.png')
        plt.imsave(output_path_image_with_pattern, image_with_pattern_rgb)
        print(f'Immagine con il Pattern {pattern_func.__name__} applicato salvata in: {output_path_image_with_pattern}')

        gray_image_with_pattern = rgb2gray(image_with_pattern)

        image_with_pattern_fft = np.fft.fftshift(np.fft.fft2(gray_image_with_pattern))

        image_with_pattern_fft_normalized = np.log(np.abs(image_with_pattern_fft) + 1)
        #image_with_pattern_fft_normalized = ((image_with_pattern_fft_normalized - image_with_pattern_fft_normalized.min()) /
        #                                   (image_with_pattern_fft_normalized.max() - image_with_pattern_fft_normalized.min()) * 255).astype(np.uint8)

        output_path_image_with_pattern_fft = os.path.join(output_folder, f'{pattern_func.__name__}_image_with_pattern_fft_{j + 1}_mode2.png')
        plt.imsave(output_path_image_with_pattern_fft, image_with_pattern_fft_normalized, cmap='viridis')
        print(f'FFT dell\'immagine con il Pattern {pattern_func.__name__} applicato salvata in: {output_path_image_with_pattern_fft}')
        '''


# Create a grid of images for visualization
fig, axs = plt.subplots(4, len(already_seen) + 1, figsize=(3 * (len(already_seen) + 1), 12))
fig.suptitle('Frequency Patterns Visualization')

# Add a column for text labels
axs[0, 0].axis('off')
axs[1, 0].axis('off')
axs[2, 0].axis('off')
axs[3, 0].axis('off')



axs[0, 0].set_title('Pattern', fontsize=12, fontweight='bold', y=0.4)
axs[1, 0].set_title('Fourier Transform\nof Pattern', fontsize=12, fontweight='bold', y=0.4)
axs[2, 0].set_title('Image with\nPattern', fontsize=12, fontweight='bold', y=0.4)
axs[3, 0].set_title('Fourier Transform\nof Image with Pattern', fontsize=12, fontweight='bold', y=0.4)

for i, pattern_name in enumerate(already_seen):
    pattern_path = os.path.join(output_folder, f'{pattern_name}_pattern_1.png')
    pattern_fft_path = os.path.join(output_folder, f'{pattern_name}_pattern_fft_1.png')
    image_with_pattern_path = os.path.join(output_folder, f'{pattern_name}_image_with_pattern_1.png')
    image_with_pattern_fft_path = os.path.join(output_folder, f'{pattern_name}_image_with_pattern_fft_1.png')

    pattern = plt.imread(pattern_path)
    pattern_fft = plt.imread(pattern_fft_path)
    image_with_pattern = plt.imread(image_with_pattern_path)
    image_with_pattern_fft = plt.imread(image_with_pattern_fft_path)
    
    # Set pattern name above each column
    axs[0, i + 1].set_title(pattern_name.replace("pattern", "").replace("_", " "), fontsize=12, fontweight='bold')

    axs[0, i + 1].imshow(pattern, cmap='gray')
    axs[1, i + 1].imshow(pattern_fft, cmap='viridis')
    axs[2, i + 1].imshow(image_with_pattern)
    axs[3, i + 1].imshow(image_with_pattern_fft, cmap='viridis')


    #for ax in axs[:, i + 1]:
        #ax.axis('off')

plt.tight_layout(rect=[0, 0, 1, 0.94])  # Adjust layout to prevent title overlap

# Save the grid as an image
output_grid_path = os.path.join(output_folder, 'patterns_overview.png')
plt.savefig(output_grid_path)
print(f'Grid image saved in: {output_grid_path}')



