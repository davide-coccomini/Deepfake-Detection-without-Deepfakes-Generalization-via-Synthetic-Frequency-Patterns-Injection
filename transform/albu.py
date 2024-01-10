import random

import cv2
import numpy as np
import torch
from albumentations import DualTransform, ImageOnlyTransform
from albumentations.augmentations.functional import crop

from torchsr.models import edsr_baseline

from skimage.color import rgb2hsv, rgb2gray, rgb2yuv
from skimage import color, exposure, transform
from skimage.exposure import equalize_hist
from albumentations import RandomCrop
from scipy.fftpack import dct, idct

def isotropically_resize_image(img, size, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC):
    h, w = img.shape[:2]

    if max(w, h) == size:
        return img
    if w > h:
        scale = size / w
        h = h * scale
        w = size
    else:
        scale = size / h
        w = w * scale
        h = size
    interpolation = interpolation_up if scale > 1 else interpolation_down

    img = img.astype('uint8')
    resized = cv2.resize(img, (int(w), int(h)), interpolation=interpolation)
    return resized


class IsotropicResize(DualTransform):
    def __init__(self, max_side, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC,
                 always_apply=False, p=1):
        super(IsotropicResize, self).__init__(always_apply, p)
        self.max_side = max_side
        self.interpolation_down = interpolation_down
        self.interpolation_up = interpolation_up

    def apply(self, img, interpolation_down=cv2.INTER_AREA, interpolation_up=cv2.INTER_CUBIC, **params):
        return isotropically_resize_image(img, size=self.max_side, interpolation_down=interpolation_down,
                                          interpolation_up=interpolation_up)

    def pattern_to_mask(self, img, **params):
        return self.apply(img, interpolation_down=cv2.INTER_NEAREST, interpolation_up=cv2.INTER_NEAREST, **params)

    def get_transform_init_args_names(self):
        return ("max_side", "interpolation_down", "interpolation_up")


class Resize4xAndBack(ImageOnlyTransform):
    def __init__(self, always_apply=False, p=0.5):
        super(Resize4xAndBack, self).__init__(always_apply, p)

    def apply(self, img, **params):
        h, w = img.shape[:2]
        scale = random.choice([2, 4])
        img = cv2.resize(img, (w // scale, h // scale), interpolation=cv2.INTER_AREA)
        img = cv2.resize(img, (w, h),
                         interpolation=random.choice([cv2.INTER_CUBIC, cv2.INTER_LINEAR, cv2.INTER_NEAREST]))
        return img


class RandomSizedCropNonEmptyMaskIfExists(DualTransform):

    def __init__(self, min_max_height, w2h_ratio=[0.7, 1.3], always_apply=False, p=0.5):
        super(RandomSizedCropNonEmptyMaskIfExists, self).__init__(always_apply, p)

        self.min_max_height = min_max_height
        self.w2h_ratio = w2h_ratio

    def apply(self, img, x_min=0, x_max=0, y_min=0, y_max=0, **params):
        cropped = crop(img, x_min, y_min, x_max, y_max)
        return cropped

    @property
    def targets_as_params(self):
        return ["mask"]

    def get_params_dependent_on_targets(self, params):
        mask = params["mask"]
        mask_height, mask_width = mask.shape[:2]
        crop_height = int(mask_height * random.uniform(self.min_max_height[0], self.min_max_height[1]))
        w2h_ratio = random.uniform(*self.w2h_ratio)
        crop_width = min(int(crop_height * w2h_ratio), mask_width - 1)
        if mask.sum() == 0:
            x_min = random.randint(0, mask_width - crop_width + 1)
            y_min = random.randint(0, mask_height - crop_height + 1)
        else:
            mask = mask.sum(axis=-1) if mask.ndim == 3 else mask
            non_zero_yx = np.argwhere(mask)
            y, x = random.choice(non_zero_yx)
            x_min = x - random.randint(0, crop_width - 1)
            y_min = y - random.randint(0, crop_height - 1)
            x_min = np.clip(x_min, 0, mask_width - crop_width)
            y_min = np.clip(y_min, 0, mask_height - crop_height)

        x_max = x_min + crop_height
        y_max = y_min + crop_width
        y_max = min(mask_height, y_max)
        x_max = min(mask_width, x_max)
        return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max}

    def get_transform_init_args_names(self):
        return "min_max_height", "height", "width", "w2h_ratio"

class CustomRandomCrop(DualTransform):
    def __init__(self, size, p=0.5) -> None:
        super(CustomRandomCrop, self).__init__(p=p)
        self.size = size
        self.prob = p

    def apply(self, img, copy=True, **params):
        if img.shape[0] < self.size or img.shape[1] < self.size:
            transform = IsotropicResize(max_side=self.size, interpolation_down=cv2.INTER_LINEAR, interpolation_up=cv2.INTER_LINEAR)
        else:
            transform = RandomCrop(self.size, self.size)
        return np.asarray(transform(image=img)["image"])

class FFT(DualTransform):
    def __init__(self, mode, p=0.5) -> None:
        super(FFT, self).__init__(p=p)
        self.prob = p
        self.mode = mode

    def apply(self, img, copy=True, **params):
        dark_image_grey_fourier = np.fft.fftshift(np.fft.fft2(rgb2gray(img)))
        mask = np.log(abs(dark_image_grey_fourier)).astype(np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))
        if self.mode == 0:
            return np.asarray(cv2.bitwise_and(img, img, mask=mask))
        else:
            mask = np.asarray(mask)
            image =  cv2.merge((mask, mask, mask))
            return image

class SR(DualTransform):
    def __init__(self, model_sr, p=0.5) -> None:
        super(SR, self).__init__(p=p)
        self.prob = p
        self.model_sr = model_sr

    def apply(self, img, copy=True, **params):
        img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)), interpolation = cv2.INTER_AREA)
        img = np.transpose(img, (2, 0, 1))
        img = torch.tensor(img, dtype=torch.float).unsqueeze(0).to(2)
        sr_img = self.model_sr(img)
        return sr_img.squeeze(0).permute(1, 2, 0).detach().cpu().numpy()


class DCT(DualTransform):
    def __init__(self, mode, p=0.5) -> None:
        super(DCT, self).__init__(p=p)
        self.prob = p
        self.mode = mode

    def rgb2gray(self, rgb):
        return cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)

    def apply(self, img, copy=True, **params):
        gray_img = self.rgb2gray(img)
        dct_coefficients = cv2.dct(cv2.dct(np.float32(gray_img), flags=cv2.DCT_ROWS), flags=cv2.DCT_ROWS)
        epsilon = 1
        mask = np.log(np.abs(dct_coefficients) + epsilon).astype(np.uint8)
        mask = cv2.resize(mask, (img.shape[1], img.shape[0]))


        if self.mode == 0:
            return cv2.bitwise_and(img, img, mask=mask)
        else:
            dct_coefficients = np.asarray(dct_coefficients)
            image = cv2.merge((dct_coefficients, dct_coefficients, dct_coefficients))
            return image





class FrequencyPatterns(DualTransform):
    def __init__(self, p=0.5) -> None:
        super(FrequencyPatterns, self).__init__(p=p)
        self.prob = p
        self.patterns = [self.pattern_grid, self.pattern_spiral, self.pattern_aura, self.pattern_mode_collapse,
                        self.pattern_circular_checkerboard, self.pattern_zigzag, self.pattern_moire, self.pattern_radial_gradient,
                        self.pattern_stripes, self.pattern_checkerboard, self.pattern_random_dots,
                        self.pattern_random_circles, self.pattern_random_diagonal_lines, self.pattern_random_ellipses,
                        self.pattern_concentric_circles, self.pattern_checkerboard,
                        self.pattern_high_frequency_noise, self.pattern_structured_noise, self.pattern_gradient_patterns]
    
    def generate_pattern(self, pattern_function, cols, rows):
        pattern = pattern_function(cols//2, rows//2)

        pattern = np.vstack([np.hstack([pattern, np.fliplr(pattern)]),
                             np.hstack([np.flipud(pattern), np.flipud(np.fliplr(pattern))])])

        return pattern
        
    def apply(self, img, required_pattern=None, return_pattern=False, copy=True, weight=0.05, mode=0, **params):
        result_channels = []
        
        if required_pattern is None:
            pattern_function = random.choice(self.patterns)
        else:
            pattern_function = required_pattern
        f_pattern = self.generate_pattern(pattern_function, cols=img.shape[1], rows=img.shape[0])

        if return_pattern:
            return f_pattern
        
        f_pattern = np.fft.fft2(f_pattern, s=(img.shape[0], img.shape[1]))

        for channel in range(img.shape[2]):
            if mode == 0:
                # Do the Fourier Transform of the channel of the image
                f_transform_channel = np.fft.fft2(img[..., channel])

                # Move to magnitude/phase representation
                magnitude_original = np.abs(f_transform_channel)
                phase_original = np.angle(f_transform_channel)

                magnitude_pattern = np.abs(f_pattern)

                # Make the weighted sum of the two magnitudes
                magnitude_result = (1 - weight) * magnitude_original + weight * magnitude_pattern

                # Combine again the obtained magnitude with the phase of the original image
                f_result_channel = magnitude_result * np.exp(1j * phase_original)

                # Make the inverse fourier transform
                result_channel = np.fft.ifft2(f_result_channel).real

                # Append the resulting channel
                result_channels.append(result_channel)
            elif mode == 1:
                # Move to magnitude/phase representation
                magnitude_pattern = np.abs(f_pattern)

                # Make the inverse fourier transform of the pattern
                pattern = np.fft.ifft2(magnitude_pattern).real

                # Combine the resulting pattern with the image in the spatial dimension
                result_channel = (1 - weight) * img[..., channel] + weight * pattern

                # Append the resulting channel
                result_channels.append(result_channel)
            elif mode == 2:
                # Do the Fourier Transform of the channel of the image
                f_transform_channel = np.fft.fft2(img[..., channel])

                # Move to magnitude/phase representation
                magnitude_original = np.abs(f_transform_channel)
                phase_original = np.angle(f_transform_channel)

                magnitude_pattern = np.abs(f_pattern)
                phase_pattern = np.angle(f_pattern)

                # Make the weighted sum of the two magnitudes
                magnitude_result = (1 - weight) * magnitude_original + weight * magnitude_pattern

                # Make the weighted sum of the two phases
                phase_result = (1 - weight) * phase_original + weight * phase_pattern

                # Combine again the obtained magnitude with the phase of the original image
                f_result_channel = magnitude_result * np.exp(1j * phase_result)

                # Make the inverse fourier transform
                result_channel = np.fft.ifft2(f_result_channel).real

                # Append the resulting channel
                result_channels.append(result_channel)



        # Stack together the channels
        result = np.stack(result_channels, axis=-1)

        # Go back to the 0-255 range
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

        return result


    def pattern_grid(self, cols, rows):
        grid_spacing = np.random.randint(5, 50)
        grid_amplitude = np.random.randint(10, 100)

        frequency_pattern = np.zeros((rows, cols))
        for x in range(0, cols, grid_spacing):
            frequency_pattern[:, x] = grid_amplitude
        for y in range(0, rows, grid_spacing):
            frequency_pattern[y, :] = grid_amplitude

        return frequency_pattern

    def pattern_spiral(self, cols, rows):
        num_turns = np.random.randint(2, 5)

        pattern = np.zeros((rows, cols))
        center_x = cols // 2
        center_y = rows // 2
        for angle in range(15, 360 * num_turns, 5):
            radius = angle / 5
            x = int(center_x + radius * np.cos(np.radians(angle)))
            y = int(center_y + radius * np.sin(np.radians(angle)))
            if 0 <= x < cols and 0 <= y < rows:
                pattern[y, x] = 255  

        return pattern

    def pattern_aura(self, cols, rows):
        aura_intensity = np.random.uniform(0.1, 0.5)
        aura_frequency = np.random.randint(5, 20)
        x = np.arange(cols)
        y = np.arange(rows)
        X, Y = np.meshgrid(x, y)

        pattern_real = np.sin(2 * np.pi * aura_frequency * (X + Y) / (cols + rows))
        pattern_imag = np.zeros_like(pattern_real)

        pattern = aura_intensity * (pattern_real + 1j * pattern_imag)

        return pattern

    def pattern_mode_collapse(self, cols, rows):
        num_repeated_patterns = np.random.randint(3, 10)
        intensity = np.random.uniform(0.1, 0.5)

        pattern = np.zeros((rows, cols))
        block_width = cols // num_repeated_patterns
        for i in range(num_repeated_patterns):
            pattern[:, i * block_width:(i + 1) * block_width] = intensity  

        return pattern

    def pattern_circular_checkerboard(self, cols, rows, num_sectors=8):
        pattern = np.zeros((rows, cols))

        center_x, center_y = cols // 2, rows // 2
        max_radius = min(center_x, center_y)
        sector_angle = 2 * np.pi / num_sectors

        for y in range(rows):
            for x in range(cols):
                dx = x - center_x
                dy = y - center_y
                angle = np.arctan2(dy, dx)

                if angle < 0:
                    angle += 2 * np.pi

                sector_index = int(angle / sector_angle)
                if sector_index % 2 == 0:
                    pattern[y, x] = 255  

        return pattern

    def pattern_zigzag(self, cols, rows):
        zigzag_amplitude = np.random.randint(10, 40)
        zigzag_frequency = np.random.randint(5, 20)

        pattern = np.zeros((rows, cols))
        for y in range(rows):
            x = int(zigzag_amplitude * np.sin(2 * np.pi * zigzag_frequency * y / rows))
            pattern[y, x:] = 255  

        return pattern

    def pattern_moire(self, cols, rows):
        num_lines = np.random.randint(5, 20)
        line_spacing = min(rows // num_lines, cols)
        intensity = np.random.uniform(0.1, 0.5)

        pattern = np.zeros((rows, cols))
        for i in range(num_lines):
            y = i * line_spacing
            if y < rows:
                pattern[y, :] = intensity  

        return pattern

    def pattern_radial_gradient(self, cols, rows):
        pattern = np.zeros((rows, cols))

        center_x, center_y = cols // 2, rows // 2
        max_distance = np.sqrt(center_x**2 + center_y**2)

        for y in range(rows):
            for x in range(cols):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                intensity = 1 - distance / max_distance
                pattern[y, x] = (255 * intensity)  

        return pattern

    def pattern_stripes(self, cols, rows):
        stripe_width = np.random.randint(5, 40)

        pattern = np.zeros((rows, cols))
        for i in range(0, rows, stripe_width):
            pattern[i:i+stripe_width, :] = 255  

        return pattern

    def pattern_checkerboard(self, cols, rows):
        square_size = np.random.randint(19, 20)

        pattern = np.zeros((rows, cols))
        for i in range(0, rows, square_size):
            for j in range(0, cols, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    pattern[i:i+square_size, j:j+square_size] = 255  

        return pattern

    def pattern_random_dots(self, cols, rows):
        num_dots = np.random.randint(10, 100)

        pattern = np.zeros((rows, cols))
        for _ in range(num_dots):
            x = np.random.randint(cols)
            y = np.random.randint(rows)
            pattern[y, x] = 255  

        return pattern



    def pattern_random_circles(self, cols, rows):
        pattern = np.zeros((rows, cols))
        num_circles = np.random.randint(5, 20)

        for _ in range(num_circles):
            x, y = np.random.randint(cols), np.random.randint(rows)
            radius = np.random.randint(5, 30)

            # Disegna manualmente il cerchio nel dominio complesso
            y_vals, x_vals = np.ogrid[-y:rows - y, -x:cols - x]
            mask = x_vals**2 + y_vals**2 <= radius**2
            circle_points = np.zeros_like(pattern)
            circle_points[mask] = 255

            # Aggiorna il pattern complesso con i valori del cerchio
            pattern += circle_points

        return pattern


    def pattern_random_diagonal_lines(self, cols, rows):
        num_lines = np.random.randint(5, 20)

        pattern = np.zeros((rows, cols))
        for _ in range(num_lines):
            thickness = np.random.randint(1, 5)
            color = 255 if np.random.rand() > 0.5 else 0  
            x1 = 0
            y1 = np.random.randint(rows)
            x2 = cols
            y2 = np.random.randint(rows)
            
            # Draw the line manually in the complex domain, considering image boundaries
            y_values = np.linspace(y1, y2, rows)
            x_values = np.linspace(x1, x2, cols)
            for i in range(min(len(x_values), len(y_values))):
                y_idx = int(np.clip(y_values[i], 0, rows - 1))
                x_idx = int(np.clip(x_values[i], 0, cols - 1))
                pattern[y_idx, x_idx] = color

        return pattern


    def pattern_random_ellipses(self, cols, rows):
        num_ellipses = np.random.randint(5, 20)

        pattern = np.zeros((rows, cols))
        for _ in range(num_ellipses):
            center_x = np.random.randint(cols)
            center_y = np.random.randint(rows)
            major_axis = np.random.randint(10, 50)
            minor_axis = np.random.randint(5, 30)
            angle = np.random.randint(0, 180)

            # Disegna manualmente l'ellisse nel dominio complesso
            t = np.linspace(15, 2 * np.pi, 100)
            x = center_x + major_axis * np.cos(np.radians(angle)) * np.cos(t) - minor_axis * np.sin(np.radians(angle)) * np.sin(t)
            y = center_y + major_axis * np.sin(np.radians(angle)) * np.cos(t) + minor_axis * np.cos(np.radians(angle)) * np.sin(t)

            # Imposta i valori dell'ellisse nel pattern complesso
            for i in range(len(x)):
                x_coord = int(round(x[i]))
                y_coord = int(round(y[i]))
                if 0 <= x_coord < cols and 0 <= y_coord < rows:
                    pattern[y_coord, x_coord] = 255  

        return pattern



    def pattern_concentric_circles(self, cols, rows):
        num_circles = np.random.randint(5, 20)

        pattern = np.zeros((rows, cols))
        center_x = cols // 2
        center_y = rows // 2
        max_radius = min(center_x, center_y) - 10

        for _ in range(num_circles):
            radius = np.random.randint(15, max_radius)

            # Disegna manualmente il cerchio nel dominio complesso
            t = np.linspace(15, 2 * np.pi, 100)
            x = center_x + radius * np.cos(t)
            y = center_y + radius * np.sin(t)

            # Imposta i valori del cerchio nel pattern complesso
            for i in range(len(x)):
                x_coord = int(round(x[i]))
                y_coord = int(round(y[i]))
                if 0 <= x_coord < cols and 0 <= y_coord < rows:
                    pattern[y_coord, x_coord] = 255  

        return pattern

    def pattern_high_frequency_noise(self, cols, rows):
        noise_std = np.random.uniform(0.005, 0.02)

        noise = np.random.normal(10, noise_std, (rows, cols))
        return noise

    def pattern_structured_noise(self, cols, rows):
        intensity = np.random.uniform(0.1, 0.5)

        pattern = np.random.rand(rows, cols) * intensity  
        return pattern

    def pattern_gradient_patterns(self, cols, rows):
        intensity = np.random.uniform(0.1, 0.5)

        x, y = np.meshgrid(np.arange(cols), np.arange(rows))
        gradient_x = x.astype(np.float32) / (cols - 1)
        gradient_y = y.astype(np.float32) / (rows - 1)
        pattern = (gradient_x + gradient_y) * intensity  
        return pattern