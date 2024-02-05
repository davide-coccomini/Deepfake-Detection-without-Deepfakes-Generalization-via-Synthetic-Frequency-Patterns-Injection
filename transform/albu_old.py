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
        self.patterns = [self.pattern_grid, self.pattern_grid, self.pattern_grid, self.pattern_aura, self.pattern_zigzag, self.pattern_checkerboard, self.pattern_mode_collapse, self.pattern_moire, self.pattern_radial_gradient, self.pattern_random_cross, self.pattern_circular_checkerboard, self.pattern_concentric_circles, self.pattern_random_circles, self.pattern_stripes, self.pattern_random_dots, self.pattern_random_lines, self.pattern_random_diagonal_lines, self.pattern_random_ellipses, self.pattern_spiral, self.pattern_checkerboard, self.pattern_gradient_patterns, self.pattern_structured_noise, self.pattern_high_frequency_noise]

    def apply(self, img, required_pattern = None, return_pattern = False, weight = 0.98, copy=True, **params):
        # Initialize an empty array to store the results
        result_channels = []

        if required_pattern != None:
            pattern = required_pattern(cols=img.shape[1], rows=img.shape[0])
        else:
            # Create a pattern with the same shape as the input image channel
            pattern = random.choice(patterns)(cols=img.shape[1], rows=img.shape[0])
        
        if return_pattern:
            return pattern
            
        for channel in range(img.shape[2]):
            # Perform FFT on the current channel
            f_transform_channel = np.fft.fft2(img[..., channel])
            
            # Perform FFT on the pattern
            f_pattern = np.fft.fft2(pattern, s=(img.shape[0], img.shape[1]))

            # Add the pattern to the transformed channel
            f_result_channel = (1 - weight) * f_transform_channel + (weight * f_pattern)

            # Perform inverse FFT on the channel
            result_channel = np.fft.ifft2(f_result_channel).real
            
            result_channels.append(result_channel)

        # Combine the individual channels into a single result
        result = np.stack(result_channels, axis=-1)

        # Normalize the result
        result = cv2.normalize(result, None, 0, 255, cv2.NORM_MINMAX)

        return result




    def pattern_grid(self, cols, rows):
        grid_spacing = np.random.randint(5, 50)
        grid_amplitude = np.random.randint(10, 100)

        pattern = np.zeros((rows, cols), dtype=np.uint8)
        for x in range(0, cols, grid_spacing):
            pattern[:, x] = grid_amplitude
        for y in range(0, rows, grid_spacing):
            pattern[y, :] = grid_amplitude
        return pattern

    
    def pattern_spiral(self, cols, rows):
        num_turns = np.random.randint(2, 5)  # Adjust the number of turns as needed

        pattern = np.zeros((rows, cols), dtype=np.uint8)
        center_x = cols // 2
        center_y = rows // 2
        for angle in range(0, 360 * num_turns, 5):
            radius = angle / 5  # Adjust the radius scaling as needed
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
        pattern = np.sin(2 * np.pi * aura_frequency * (X + Y) / (cols + rows))
        pattern = (pattern * aura_intensity * 255).astype(np.uint8)
        return pattern

    def pattern_mode_collapse(self, cols, rows):
        num_repeated_patterns = np.random.randint(3, 10)
        intensity = np.random.uniform(0.1, 0.5)

        pattern = torch.zeros((rows, cols), dtype=torch.float32)
        block_width = cols // num_repeated_patterns
        for i in range(num_repeated_patterns):
            pattern[:, i * block_width:(i + 1) * block_width] = intensity

        return pattern

    def pattern_circular_checkerboard(self, cols, rows, num_sectors=8):
        # Create an empty pattern
        pattern = np.zeros((rows, cols), dtype=np.uint8)

        # Determine the center of the pattern
        center_x, center_y = cols // 2, rows // 2

        # Calculate the radius of the circular checkerboard
        max_radius = min(center_x, center_y)

        # Calculate the angular size of each sector
        sector_angle = 2 * np.pi / num_sectors

        # Create the circular checkerboard
        for y in range(rows):
            for x in range(cols):
                dx = x - center_x
                dy = y - center_y
                angle = np.arctan2(dy, dx)  # Calculate the angle from the center

                if angle < 0:
                    angle += 2 * np.pi  # Adjust for negative angles

                sector_index = int(angle / sector_angle)
                if sector_index % 2 == 0:
                    pattern[y, x] = 255

        return pattern

    def pattern_zigzag(self, cols, rows):
        zigzag_amplitude = np.random.randint(10, 40)
        zigzag_frequency = np.random.randint(5, 20)

        pattern = np.zeros((rows, cols), dtype=np.uint8)
        for y in range(rows):
            x = int(zigzag_amplitude * np.sin(2 * np.pi * zigzag_frequency * y / rows))
            pattern[y, x:] = 255
        return pattern

    def pattern_moire(self, cols, rows):
        num_lines = np.random.randint(5, 20)
        # Ensure that line spacing is within bounds
        line_spacing = min(rows // num_lines, cols)  # Adjust as needed
        intensity = np.random.uniform(0.1, 0.5)

        pattern = torch.zeros((rows, cols), dtype=torch.float32)
        for i in range(num_lines):
            y = i * line_spacing
            if y < rows:
                pattern[y, :] = intensity

        return pattern

    def pattern_radial_gradient(self, cols, rows):
        # Create an empty pattern
        pattern = np.zeros((rows, cols), dtype=np.uint8)

        # Determine the center of the pattern
        center_x, center_y = cols // 2, rows // 2

        # Calculate the maximum distance from the center
        max_distance = np.sqrt(center_x**2 + center_y**2)

        # Generate a radial gradient
        for y in range(rows):
            for x in range(cols):
                distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                intensity = int(255 * (1 - distance / max_distance))
                pattern[y, x] = intensity

        return pattern


    def pattern_stripes(self, cols, rows):
        stripe_width = np.random.randint(5, 40)  # Adjust the width of the stripes as needed

        pattern = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(0, rows, stripe_width):
            pattern[i:i+stripe_width, :] = 255
        return pattern


    def pattern_checkerboard(self, cols, rows):
        square_size = np.random.randint(5, 40)  # Adjust the size of the squares as needed

        pattern = np.zeros((rows, cols), dtype=np.uint8)
        for i in range(0, rows, square_size):
            for j in range(0, cols, square_size):
                if (i // square_size + j // square_size) % 2 == 0:
                    pattern[i:i+square_size, j:j+square_size] = 255
        return pattern


    def pattern_random_dots(self, cols, rows):
        num_dots = np.random.randint(10, 100)  # Adjust the number of dots as needed
        pattern = np.zeros((rows, cols), dtype=np.uint8)
        for _ in range(num_dots):
            x = np.random.randint(cols)
            y = np.random.randint(rows)
            pattern[y, x] = 255
        return pattern



    def pattern_random_lines(self, cols, rows):
        num_lines = np.random.randint(5, 20)  # Adjust the number of lines as needed
        pattern = np.zeros((rows, cols), dtype=np.uint8)
        for _ in range(num_lines):
            x1 = np.random.randint(cols)
            y1 = np.random.randint(rows)
            x2 = np.random.randint(cols)
            y2 = np.random.randint(rows)
            cv2.line(pattern, (x1, y1), (x2, y2), 255, 1)
        return pattern

    def pattern_random_circles(self, cols, rows):
        num_circles = np.random.randint(5, 20)  # Adjust the number of circles as needed

        pattern = np.zeros((rows, cols), dtype=np.uint8)
        for _ in range(num_circles):
            x = np.random.randint(cols)
            y = np.random.randint(rows)
            radius = np.random.randint(5, 30)  # Adjust the radius range as needed
            cv2.circle(pattern, (x, y), radius, 255, -1)
        return pattern


    def pattern_random_diagonal_lines(self, cols, rows):
        num_lines = np.random.randint(5, 20)  # Adjust the number of lines as needed
        pattern = np.zeros((rows, cols), dtype=np.uint8)
        for _ in range(num_lines):
            thickness = np.random.randint(1, 5)  # Adjust the line thickness as needed
            color = 255 if np.random.rand() > 0.5 else 0
            x1 = 0
            y1 = np.random.randint(rows)
            x2 = cols
            y2 = np.random.randint(rows)
            cv2.line(pattern, (x1, y1), (x2, y2), color, thickness)
        return pattern

    def pattern_random_ellipses(self, cols, rows):
        num_ellipses = np.random.randint(5, 20)  # Adjust the number of ellipses as needed
        pattern = np.zeros((rows, cols), dtype=np.uint8)
        for _ in range(num_ellipses):
            center_x = np.random.randint(cols)
            center_y = np.random.randint(rows)
            major_axis = np.random.randint(10, 50)  # Adjust the size range as needed
            minor_axis = np.random.randint(5, 30)   # Adjust the size range as needed
            angle = np.random.randint(0, 180)       # Adjust the angle range as needed
            cv2.ellipse(pattern, (center_x, center_y), (major_axis, minor_axis), angle, 0, 360, 255, -1)
        return pattern



    def pattern_concentric_circles(self, cols, rows):
        num_circles = np.random.randint(5, 20)  # Adjust the number of circles as needed

        pattern = np.zeros((rows, cols), dtype=np.uint8)
        center_x = cols // 2
        center_y = rows // 2
        max_radius = min(center_x, center_y) - 10  # Adjust the maximum radius as needed
        for _ in range(num_circles):
            radius = np.random.randint(5, max_radius)
            cv2.circle(pattern, (center_x, center_y), radius, 255, 1)
        return pattern


    def pattern_random_cross(self, cols, rows):
        # Create an empty pattern
        pattern = np.zeros((rows, cols), dtype=np.uint8)

        # Determine the center of the pattern
        center_x, center_y = cols // 2, rows // 2

        # Define the number of rays
        num_rays = np.random.randint(4, 12)  # Adjust the range as needed

        # Generate random rays with varying lengths and intensity
        for _ in range(num_rays):
            # Randomly select an angle for the ray
            angle = np.random.uniform(0, 2 * np.pi)

            # Calculate the length of the ray (up to half the image dimensions)
            max_length = min(center_x, center_y) // 2
            ray_length = np.random.randint(10, max_length)  # Adjust the range as needed

            # Calculate the endpoint of the ray
            end_x = int(center_x + ray_length * np.cos(angle))
            end_y = int(center_y + ray_length * np.sin(angle))

            # Randomly assign intensity (brightness) to the ray
            intensity = np.random.randint(100, 256)  # Adjust the range as needed

            # Draw the ray
            cv2.line(pattern, (center_x, center_y), (end_x, end_y), intensity, 1)

        return pattern


    def pattern_checkerboard(self, cols, rows):
        block_size = np.random.randint(5, 15)
        intensity = np.random.uniform(0.1, 0.5)

        pattern = torch.zeros((rows, cols), dtype=torch.float32)
        for y in range(0, rows, block_size):
            for x in range(0, cols, block_size):
                if (x // block_size + y // block_size) % 2 == 0:
                    pattern[y:y+block_size, x:x+block_size] = intensity

        return pattern

    def pattern_high_frequency_noise(self, cols, rows):
        noise_std = np.random.uniform(0.005, 0.02)

        noise = torch.randn((rows, cols), dtype=torch.float32) * noise_std
        return noise


    def pattern_structured_noise(self, cols, rows):
        intensity = np.random.uniform(0.1, 0.5)

        pattern = torch.rand((rows, cols), dtype=torch.float32) * intensity
        return pattern

    def pattern_gradient_patterns(self, cols, rows):
        intensity = np.random.uniform(0.1, 0.5)

        x, y = torch.meshgrid(torch.arange(cols), torch.arange(rows))
        gradient_x = x.float() / (cols - 1)
        gradient_y = y.float() / (rows - 1)
        pattern = (gradient_x + gradient_y) * intensity
        return pattern
