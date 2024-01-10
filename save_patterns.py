import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from transform.albu import FrequencyPatterns

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

image_path = 'patterns/person.jpg'
base_image = cv2.imread(image_path, cv2.IMREAD_COLOR)

output_folder = 'patterns_fourier'
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

for i, pattern_func in enumerate(freq_patterns.patterns):
    for j in range(1):  
        if pattern_func.__name__ != "pattern_checkerboard":
            continue

        
        pattern = freq_patterns.apply(img=base_image, required_pattern=pattern_func, return_pattern=True, mode=0)
        pattern = pattern.astype(np.uint8)
        pattern_rgb = cv2.cvtColor(pattern, cv2.COLOR_GRAY2RGB)

        output_pattern = os.path.join(output_folder, f'{pattern_func.__name__}_pattern_{j + 1}.png')
        plt.imsave(output_pattern, pattern_rgb)
        print(f'Pattern {pattern_func.__name__} applicato salvato in: {output_pattern}')


        f_pattern = np.fft.fftshift(np.fft.fft2(pattern))
        f_pattern = np.log(np.abs(f_pattern) + 1)

        output_f_pattern = os.path.join(output_folder, f'{pattern_func.__name__}_pattern_fft_{j + 1}.png')
        plt.imsave(output_f_pattern, f_pattern, cmap='viridis')
        print(f'FFT Pattern {pattern_func.__name__} applicato salvato in: {output_f_pattern}')

        image_with_pattern = freq_patterns.apply(img=base_image, required_pattern=pattern_func, return_pattern=False, mode=0)

        image_with_pattern = image_with_pattern.astype(np.uint8)

        image_with_pattern_rgb = cv2.cvtColor(image_with_pattern, cv2.COLOR_BGR2RGB)
            
        output_path_image_with_pattern = os.path.join(output_folder, f'{pattern_func.__name__}_image_with_pattern_{j + 1}.png')
        plt.imsave(output_path_image_with_pattern, image_with_pattern_rgb)
        print(f'Immagine con il Pattern {pattern_func.__name__} applicato salvata in: {output_path_image_with_pattern}')

        gray_image_with_pattern = rgb2gray(image_with_pattern)

        image_with_pattern_fft = np.fft.fftshift(np.fft.fft2(gray_image_with_pattern))

        image_with_pattern_fft_normalized = np.log(np.abs(image_with_pattern_fft) + 1)
        #image_with_pattern_fft_normalized = ((image_with_pattern_fft_normalized - image_with_pattern_fft_normalized.min()) /
        #                                   (image_with_pattern_fft_normalized.max() - image_with_pattern_fft_normalized.min()) * 255).astype(np.uint8)

        output_path_image_with_pattern_fft = os.path.join(output_folder, f'{pattern_func.__name__}_image_with_pattern_fft_{j + 1}.png')
        plt.imsave(output_path_image_with_pattern_fft, image_with_pattern_fft_normalized, cmap='viridis')
        print(f'FFT dell\'immagine con il Pattern {pattern_func.__name__} applicato salvata in: {output_path_image_with_pattern_fft}')


        
        image_with_pattern = freq_patterns.apply(img=base_image, required_pattern=pattern_func, return_pattern=False, mode=1)

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




        image_with_pattern = freq_patterns.apply(img=base_image, required_pattern=pattern_func, return_pattern=False, mode=2)

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