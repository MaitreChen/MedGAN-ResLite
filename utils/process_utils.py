import os

from PIL import Image
import cv2

import torchvision.transforms as transforms


def center_crop_images(input_dir, output_dir, target_size, crop_size):
    """
    Perform center cropping on images in the input directory and save the cropped images to the output directory.

    Args:
        input_dir (str): Input directory containing image files to be processed.
        output_dir (str): Output directory to save the cropped image files.
        target_size (tuple): Target size (width, height) to resize images while maintaining aspect ratio.
        crop_size (int or tuple): Crop size, either an integer or a tuple (width, height). If an integer, width and height are the same.

    Returns:
        None
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Iterate through image files in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith(('.jpg', '.jpeg', '.png')):
            # Read the image
            img_path = os.path.join(input_dir, filename)
            img = cv2.imread(img_path)

            # Resize the image while maintaining aspect ratio
            h, w = img.shape[:2]
            if h > w:
                new_h = target_size[1]
                new_w = int(w * (new_h / h))
            else:
                new_w = target_size[0]
                new_h = int(h * (new_w / w))
            img = cv2.resize(img, (new_w, new_h))

            # Calculate crop position
            if isinstance(crop_size, int):
                crop_width = crop_size
                crop_height = crop_size
            else:
                crop_width, crop_height = crop_size
            start_x = max(new_w // 2 - crop_width // 2, 0)
            start_y = max(new_h // 2 - crop_height // 2, 0)

            # Crop the image
            cropped_img = img[start_y:start_y + crop_height, start_x:start_x + crop_width]

            # Save the cropped image
            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, cropped_img)
    print("Successfully!")


def resize_images(input_folder, output_folder, target_size=(224, 224)):
    """
    Resize all images in the specified folder to the specified size and save them to the target folder.

    Args:
        input_folder (str): Path to the input folder.
        output_folder (str): Path to the output folder.
        target_size (tuple): Target size, in the form of (width, height). Default is (224, 224).
    """
    # Create the output folder
    os.makedirs(output_folder, exist_ok=True)

    # Define the transform
    resize_transform = transforms.Resize(target_size)

    # Iterate over image files and resize
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith('.jpeg') or file.endswith('.png') or file.endswith('.jpg'):
                # Read the image
                image_path = os.path.join(root, file)
                image = Image.open(image_path)

                print("Original image size:", image.size)

                # Apply the transform
                resized_image = resize_transform(image)

                print("Resized image size:", resized_image.size)

                # Define the output path
                output_path = os.path.join(output_folder, os.path.relpath(root, input_folder), file)

                # Create the output folder
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # Save the resized image
                resized_image.save(output_path)

    print("Image resizing completed.")


def merge_image_folders(train_folder, val_folder, target_folder):
    import os
    import shutil
    """
    Merge image folders from train_folder and val_folder into target_folder.

    Args:
        train_folder (str): Path to the train folder.
        val_folder (str): Path to the val folder.
        target_folder (str): Path to the target folder where images will be merged.
    """
    # Create target folder if it doesn't exist
    os.makedirs(target_folder, exist_ok=True)

    # Copy images from train folder
    for filename in os.listdir(train_folder):
        src = os.path.join(train_folder, filename)
        dst = os.path.join(target_folder, filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)

    # Copy images from val folder
    for filename in os.listdir(val_folder):
        src = os.path.join(val_folder, filename)
        dst = os.path.join(target_folder, filename)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copyfile(src, dst)

    print("Files successfully copied to the target folder.")


if __name__ == '__main__':
    #     project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    #
    #     # preprocess 1
    #     crop_size = (64, 64)
    #     target_size = (64, 64)
    #
    #     for classify_dir in ['pneumonia', 'normal']:
    #         for mode in ['train','test']:
    #             input_dir = os.path.join(project_root, "data", "chest_xray2017", mode, classify_dir)
    #             output_dir = os.path.join(project_root, "data", "chest_xray2017_processed", mode, classify_dir)
    #
    #             center_crop_images(input_dir, output_dir, target_size, crop_size)
    #
    # merge train and val normal data for fid
    train_normal_folder = '../data/real/train/normal'
    val_normal_folder = '../data/real/val/normal'
    target_folder = '../data/real_valid_normal_images'

    merge_image_folders(train_normal_folder, val_normal_folder, target_folder)

#     # # preprocess for fid
#     # input_folder = '../data/chest_xray2017_size64_train_val_merge'
#     # output_folder = '../data/chest_xray2017_size64_train_val_merge'
#     # resize_images(input_folder, output_folder)
