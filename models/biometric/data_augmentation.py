"""
Biometric-specific data augmentation for palm vein recognition.
Carefully designed to preserve vein patterns while improving model robustness.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import numpy as np
import random
from PIL import Image, ImageFilter
import cv2
from typing import Tuple, List, Optional


class BiometricAugmentation:
    """
    Base class for biometric-aware augmentations that preserve critical vein patterns.
    """
    
    def __init__(self, preserve_patterns: bool = True):
        self.preserve_patterns = preserve_patterns
    
    def __call__(self, image):
        raise NotImplementedError


class CarefulRotation(BiometricAugmentation):
    """
    Rotation augmentation with limited angles to preserve vein orientation.
    Vein patterns are sensitive to large rotations.
    """
    
    def __init__(self, max_angle: float = 10.0, probability: float = 0.5):
        super().__init__()
        self.max_angle = max_angle
        self.probability = probability
    
    def __call__(self, image):
        if random.random() < self.probability:
            angle = random.uniform(-self.max_angle, self.max_angle)
            return TF.rotate(image, angle, interpolation=Image.BILINEAR, fill=0)
        return image


class CarefulTranslation(BiometricAugmentation):
    """
    Small translation augmentation to simulate slight palm positioning variations.
    """
    
    def __init__(self, max_translate_percent: float = 0.1, probability: float = 0.5):
        super().__init__()
        self.max_translate = max_translate_percent
        self.probability = probability
    
    def __call__(self, image):
        if random.random() < self.probability:
            width, height = image.size
            max_dx = int(width * self.max_translate)
            max_dy = int(height * self.max_translate)
            
            dx = random.randint(-max_dx, max_dx)
            dy = random.randint(-max_dy, max_dy)
            
            return TF.affine(image, angle=0, translate=(dx, dy), scale=1, shear=0)
        return image


class GaussianNoise(BiometricAugmentation):
    """
    Add Gaussian noise to simulate sensor noise and improve robustness.
    """
    
    def __init__(self, std: float = 0.02, probability: float = 0.3):
        super().__init__()
        self.std = std
        self.probability = probability
    
    def __call__(self, image):
        if random.random() < self.probability:
            # Convert PIL to tensor for noise addition
            if isinstance(image, Image.Image):
                tensor = TF.to_tensor(image)
            else:
                tensor = image
            
            noise = torch.randn_like(tensor) * self.std
            noisy_tensor = torch.clamp(tensor + noise, 0, 1)
            
            # Convert back to PIL if input was PIL
            if isinstance(image, Image.Image):
                return TF.to_pil_image(noisy_tensor)
            else:
                return noisy_tensor
        return image


class ContrastAdjustment(BiometricAugmentation):
    """
    Careful contrast adjustment to simulate different lighting conditions.
    """
    
    def __init__(self, contrast_range: Tuple[float, float] = (0.8, 1.2), probability: float = 0.4):
        super().__init__()
        self.contrast_range = contrast_range
        self.probability = probability
    
    def __call__(self, image):
        if random.random() < self.probability:
            factor = random.uniform(*self.contrast_range)
            return TF.adjust_contrast(image, factor)
        return image


class BrightnessAdjustment(BiometricAugmentation):
    """
    Mild brightness adjustment for lighting variations.
    """
    
    def __init__(self, brightness_range: Tuple[float, float] = (0.9, 1.1), probability: float = 0.4):
        super().__init__()
        self.brightness_range = brightness_range
        self.probability = probability
    
    def __call__(self, image):
        if random.random() < self.probability:
            factor = random.uniform(*self.brightness_range)
            return TF.adjust_brightness(image, factor)
        return image


class CarefulSharpening(BiometricAugmentation):
    """
    Mild sharpening to enhance vein visibility slightly.
    """
    
    def __init__(self, sharpness_range: Tuple[float, float] = (1.0, 1.3), probability: float = 0.2):
        super().__init__()
        self.sharpness_range = sharpness_range
        self.probability = probability
    
    def __call__(self, image):
        if random.random() < self.probability:
            factor = random.uniform(*self.sharpness_range)
            return image.filter(ImageFilter.UnsharpMask(radius=1, percent=int((factor-1)*100), threshold=0))
        return image


class MildBlur(BiometricAugmentation):
    """
    Very mild blur to simulate slight camera defocus.
    """
    
    def __init__(self, blur_radius: float = 0.5, probability: float = 0.15):
        super().__init__()
        self.blur_radius = blur_radius
        self.probability = probability
    
    def __call__(self, image):
        if random.random() < self.probability:
            return image.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))
        return image


class RandomScale(BiometricAugmentation):
    """
    Small scale changes to simulate distance variations.
    """
    
    def __init__(self, scale_range: Tuple[float, float] = (0.95, 1.05), probability: float = 0.3):
        super().__init__()
        self.scale_range = scale_range
        self.probability = probability
    
    def __call__(self, image):
        if random.random() < self.probability:
            scale = random.uniform(*self.scale_range)
            width, height = image.size
            new_width = int(width * scale)
            new_height = int(height * scale)
            
            # Resize and crop/pad to original size
            resized = image.resize((new_width, new_height), Image.BILINEAR)
            
            if scale > 1.0:  # Crop if enlarged
                left = (new_width - width) // 2
                top = (new_height - height) // 2
                return resized.crop((left, top, left + width, top + height))
            else:  # Pad if shrunk
                new_image = Image.new('L', (width, height), 0)
                paste_x = (width - new_width) // 2
                paste_y = (height - new_height) // 2
                new_image.paste(resized, (paste_x, paste_y))
                return new_image
        return image


class RandomPerspectiveTransformation(BiometricAugmentation):
    """
    Random Perspective Transformation (RPT) as described in RSNet paper.
    Paper: RSNet (IEEE TIFS 2025), Section IV-B
    
    Simulates viewpoint changes by applying random perspective distortion.
    This helps the model become robust to pose variations during palm capture.
    
    Args:
        distortion_scale (float): Controls the degree of distortion. 
            Paper refers to this as parameter 'r'. Higher values = more distortion.
            Recommended range: 0.1-0.3 for palm vein images.
        probability (float): Probability of applying the transform (p_RPT in paper).
    """
    
    def __init__(self, distortion_scale: float = 0.2, probability: float = 0.5):
        super().__init__()
        self.distortion_scale = distortion_scale
        self.probability = probability
    
    def __call__(self, image):
        if random.random() < self.probability:
            width, height = image.size
            
            # Calculate distortion offsets based on distortion_scale
            half_height = height // 2
            half_width = width // 2
            
            # Random offsets for the 4 corners
            distort = int(min(height, width) * self.distortion_scale)
            
            # Define source points (full image corners)
            # top-left, top-right, bottom-right, bottom-left
            src_pts = [
                (0, 0),
                (width, 0),
                (width, height),
                (0, height)
            ]
            
            # Define destination points with random perspective distortion
            dst_pts = [
                (random.randint(0, distort), random.randint(0, distort)),
                (width - random.randint(0, distort), random.randint(0, distort)),
                (width - random.randint(0, distort), height - random.randint(0, distort)),
                (random.randint(0, distort), height - random.randint(0, distort))
            ]
            
            # Use torchvision's perspective transform
            return TF.perspective(image, src_pts, dst_pts, interpolation=Image.BILINEAR, fill=0)
        return image


class RandomGammaAdjustment(BiometricAugmentation):
    """
    Random Gamma Adjustment (RGA) as described in RSNet paper.
    Paper: RSNet (IEEE TIFS 2025), Section IV-B
    
    Applies gamma correction to simulate different illumination conditions.
    Gamma < 1: brightens the image (enhances dark regions)
    Gamma > 1: darkens the image (enhances bright regions)
    
    Args:
        gamma_range (tuple): Range of gamma values (min, max).
            Paper refers to this as parameter 'γ'. 
            Recommended range: (0.7, 1.3) for palm vein images.
        probability (float): Probability of applying the transform (p_RGA in paper).
    """
    
    def __init__(self, gamma_range: Tuple[float, float] = (0.7, 1.3), probability: float = 0.5):
        super().__init__()
        self.gamma_range = gamma_range
        self.probability = probability
    
    def __call__(self, image):
        if random.random() < self.probability:
            gamma = random.uniform(*self.gamma_range)
            return TF.adjust_gamma(image, gamma)
        return image


class BiometricCompose:
    """
    Custom compose class for biometric augmentations with probability control.
    """
    
    def __init__(self, transforms: List[BiometricAugmentation], overall_probability: float = 0.8):
        self.transforms = transforms
        self.overall_probability = overall_probability
    
    def __call__(self, image):
        if random.random() < self.overall_probability:
            for t in self.transforms:
                image = t(image)
        return image


class TestTimeAugmentation:
    """
    Test-time augmentation for improved inference robustness.
    """
    
    def __init__(self, n_augmentations: int = 5):
        self.n_augmentations = n_augmentations
        self.augmentations = [
            CarefulRotation(max_angle=5.0, probability=1.0),
            CarefulTranslation(max_translate_percent=0.05, probability=1.0),
            ContrastAdjustment(contrast_range=(0.95, 1.05), probability=1.0),
            BrightnessAdjustment(brightness_range=(0.98, 1.02), probability=1.0),
        ]
    
    def __call__(self, image):
        """
        Generate multiple augmented versions of the image.
        
        Returns:
            List of augmented images (including original)
        """
        augmented_images = [image]  # Include original
        
        for _ in range(self.n_augmentations):
            aug_image = image
            for aug in self.augmentations:
                if random.random() < 0.3:  # Lower probability for TTA
                    aug_image = aug(aug_image)
            augmented_images.append(aug_image)
        
        return augmented_images


def get_training_transforms(image_size: Tuple[int, int] = (128, 128)):
    """
    Get training transforms for palm vein recognition.
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Composed transforms for training
    """
    biometric_augs = BiometricCompose([
        CarefulRotation(max_angle=10.0, probability=0.5),
        CarefulTranslation(max_translate_percent=0.1, probability=0.5),
        GaussianNoise(std=0.02, probability=0.3),
        ContrastAdjustment(contrast_range=(0.8, 1.2), probability=0.4),
        BrightnessAdjustment(brightness_range=(0.9, 1.1), probability=0.4),
        CarefulSharpening(sharpness_range=(1.0, 1.3), probability=0.2),
        MildBlur(blur_radius=0.5, probability=0.15),
        RandomScale(scale_range=(0.95, 1.05), probability=0.3),
    ], overall_probability=0.8)
    
    standard_transforms = transforms.Compose([
        transforms.Resize(image_size),
        biometric_augs,
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Normalize to [-1, 1]
    ])
    
    return standard_transforms


def get_validation_transforms(image_size: Tuple[int, int] = (128, 128)):
    """
    Get validation transforms (no augmentation).
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Composed transforms for validation
    """
    return transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ])


def get_test_transforms(image_size: Tuple[int, int] = (128, 128)):
    """
    Get test transforms (same as validation).
    
    Args:
        image_size: Target image size (height, width)
    
    Returns:
        Composed transforms for testing
    """
    return get_validation_transforms(image_size)


def get_tta_transforms(image_size: Tuple[int, int] = (128, 128), n_augmentations: int = 5):
    """
    Get test-time augmentation transforms.
    
    Args:
        image_size: Target image size (height, width)
        n_augmentations: Number of augmented versions to generate
    
    Returns:
        TTA transform function
    """
    tta = TestTimeAugmentation(n_augmentations=n_augmentations)
    
    def tta_transform(image):
        # Apply TTA to get multiple versions
        augmented_images = tta(image)
        
        # Convert all to tensors and normalize
        tensor_images = []
        base_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        
        for img in augmented_images:
            tensor_images.append(base_transform(img))
        
        return torch.stack(tensor_images)
    
    return tta_transform


class AugmentationStrengthScheduler:
    """
    Scheduler to adjust augmentation strength during training.
    Reduce augmentation as model becomes more confident.
    """
    
    def __init__(self, initial_strength: float = 1.0, final_strength: float = 0.5):
        self.initial_strength = initial_strength
        self.final_strength = final_strength
        self.current_strength = initial_strength
    
    def step(self, epoch: int, total_epochs: int, validation_accuracy: float):
        """Update augmentation strength based on training progress."""
        # Linear decay based on epochs
        progress = epoch / total_epochs
        epoch_factor = 1.0 - progress
        
        # Accuracy-based adjustment (reduce augmentation for high accuracy)
        accuracy_factor = max(0.5, 1.0 - validation_accuracy)
        
        # Combined strength
        self.current_strength = (
            self.initial_strength * epoch_factor * accuracy_factor +
            self.final_strength * (1 - epoch_factor)
        )
        
        return self.current_strength
    
    def get_current_transforms(self, image_size: Tuple[int, int] = (128, 128)):
        """Get transforms with current augmentation strength."""
        # Scale probabilities by current strength
        strength = self.current_strength
        
        biometric_augs = BiometricCompose([
            CarefulRotation(max_angle=10.0 * strength, probability=0.5 * strength),
            CarefulTranslation(max_translate_percent=0.1 * strength, probability=0.5 * strength),
            GaussianNoise(std=0.02 * strength, probability=0.3 * strength),
            ContrastAdjustment(
                contrast_range=(1.0 - 0.2 * strength, 1.0 + 0.2 * strength), 
                probability=0.4 * strength
            ),
            BrightnessAdjustment(
                brightness_range=(1.0 - 0.1 * strength, 1.0 + 0.1 * strength), 
                probability=0.4 * strength
            ),
            RandomScale(
                scale_range=(1.0 - 0.05 * strength, 1.0 + 0.05 * strength), 
                probability=0.3 * strength
            ),
        ], overall_probability=0.8 * strength)
        
        return transforms.Compose([
            transforms.Resize(image_size),
            biometric_augs,
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])


if __name__ == "__main__":
    # Test augmentations
    from PIL import Image
    import matplotlib.pyplot as plt
    
    print("Testing biometric augmentations...")
    
    # Create a simple test image (simulating vein pattern)
    test_image = Image.new('L', (128, 128), 0)
    # Add some "vein-like" patterns (simple lines)
    import PIL.ImageDraw as ImageDraw
    draw = ImageDraw.Draw(test_image)
    draw.line([(20, 30), (100, 80)], fill=255, width=2)
    draw.line([(50, 20), (80, 100)], fill=255, width=2)
    draw.line([(10, 60), (120, 70)], fill=255, width=2)
    
    # Test training transforms
    train_transforms = get_training_transforms()
    
    # Apply augmentations multiple times to see variation
    print("Applying training transforms...")
    augmented_images = []
    for i in range(5):
        aug_img = train_transforms(test_image)
        augmented_images.append(aug_img)
    
    print(f"Generated {len(augmented_images)} augmented images")
    print(f"Image tensor shape: {augmented_images[0].shape}")
    print(f"Image value range: [{augmented_images[0].min():.3f}, {augmented_images[0].max():.3f}]")
    
    # Test TTA
    print("Testing TTA transforms...")
    tta_transforms = get_tta_transforms(n_augmentations=3)
    tta_batch = tta_transforms(test_image)
    print(f"TTA batch shape: {tta_batch.shape}")
    
    print("Augmentation tests completed successfully!")






