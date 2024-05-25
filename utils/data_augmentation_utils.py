import random
import numpy as np

from PIL import Image


class ShuffleImageRegions(object):
    """
    Shuffle regions of an image.

    Args:
        region_size (int): Size of each region.

    Returns:
        PIL.Image: Image with shuffled regions.

    """

    def __init__(self):
        # self.region_size = region_size
        pass

    def __call__(self, image):
        # Convert PIL Image object to NumPy array
        image_array = np.array(image)

        # Get the dimensions of the image
        height, width = image_array.shape[:2]

        # Divide the image into four regions
        top_left = image_array[:height // 2, :width // 2]
        top_right = image_array[:height // 2, width // 2:]
        bottom_left = image_array[height // 2:, :width // 2]
        bottom_right = image_array[height // 2:, width // 2:]

        # Shuffle each region
        regions = [top_left, top_right, bottom_left, bottom_right]
        np.random.shuffle(regions)

        # Merge the four regions
        shuffled_image = np.vstack((np.hstack((regions[0], regions[1])),
                                    np.hstack((regions[2], regions[3]))))

        shuffled_image = Image.fromarray(shuffled_image)

        return shuffled_image


class RandomErasing(object):
    """
    Apply random erasing augmentation to images.

    Args:
        p (float): Probability of applying the augmentation. Default is 0.5.
        scale (tuple): Range of values to scale the area of erased patch. Default is (0.02, 0.4).
        ratio (tuple): Range of values to scale the aspect ratio of erased patch. Default is (0.3, 3).

    Returns:
        PIL.Image: Image with random erasing augmentation applied.

    """

    def __init__(self, p=0.5, scale=(0.02, 0.4), ratio=(0.3, 3)):
        self.p = p
        self.scale = scale
        self.ratio = ratio

    def __call__(self, img):
        if random.uniform(0, 1) >= self.p:
            return img

        width, height = img.size
        area = width * height

        for attempt in range(100):
            target_area = random.uniform(*self.scale) * area
            aspect_ratio = random.uniform(*self.ratio)

            h = int(round((target_area * aspect_ratio) ** 0.5))
            w = int(round((target_area / aspect_ratio) ** 0.5))

            if w < width and h < height:
                x1 = random.randint(0, width - w)
                y1 = random.randint(0, height - h)
                img.paste(0, (x1, y1, x1 + w, y1 + h))  # Paste black patch over the selected region
                return img

        return img


class RandomGaussianNoise(object):
    """
    Apply random Gaussian noise to images.

    Args:
        p (float): Probability of applying the noise. Default is 0.5.
        mean (float): Mean of the Gaussian distribution. Default is 0.
        std (float): Standard deviation of the Gaussian distribution. Default is 10.

    Returns:
        PIL.Image: Image with added Gaussian noise.

    """

    def __init__(self, p=0.5, mean=0, std=10):
        self.p = p
        self.mean = mean
        self.std = std

    def __call__(self, img):
        if random.uniform(0, 1) >= self.p:
            return img

        img_array = np.array(img)  # Convert PIL Image object to NumPy array
        noise = np.random.normal(self.mean, self.std, img_array.shape).astype(np.uint8)
        noisy_img = img_array + noise  # Add noise directly to image
        return Image.fromarray(noisy_img)  # Convert NumPy array back to PIL Image object


class RandomResizedCrop(object):
    def __init__(self, size, scale=(0.6, 1.2)):
        """
        Initialize RandomResizedCrop transform.

        Args:
            size (int): Desired output size.
            scale (tuple): Range of the random crop scale.
        """
        self.size = size
        self.scale = scale

    def __call__(self, image):
        # Random crop size
        crop_size = int(random.uniform(self.size * self.scale[0], self.size * self.scale[1]))

        # Random crop position
        left = random.randint(0, image.width - crop_size-1>0)
        top = random.randint(0, image.height - crop_size-1>0)

        # Perform crop
        image = image.crop((left, top, left + crop_size, top + crop_size))
        image = image.resize((self.size, self.size), Image.BILINEAR)

        return image

