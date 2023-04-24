import os
import random
import shutil


def print_info():
    print('-------------')


def create_folder(path: str):
    """
    This function creates a folder at the specified path if it does not already exist
    """
    if not os.path.exists(path):
        os.makedirs(path)


def copy_files_to_output_dir(files, dataset_dir, output_dir, sub_dir):
    for file in files:
        src = os.path.join(dataset_dir, file)
        dst = os.path.join(output_dir, sub_dir, file)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        shutil.copy(src, dst)


def split_dataset(dataset_dir: str, output_dir: str, train_percent: float, val_percent: float, test_percent: float):
    """
    Split the dataset into train, validation, and test sets and copy them to the output directory.

    Args:
    dataset_dir: The path to the dataset directory.
    output_dir: The path to the output directory.
    train_percent: The percentage of data to use for training.
    val_percent: The percentage of data to use for validation.
    test_percent: The percentage of data to use for testing.
    """
    # Create the output directory if it doesn't exist
    create_folder(output_dir)

    # Get a list of all the files in the dataset directory
    files = os.listdir(dataset_dir)

    # Shuffle the files randomly
    random.shuffle(files)

    # Calculate the number of files for each split
    num_train = int(len(files) * train_percent)
    num_val = int(len(files) * val_percent)
    num_test = int(len(files) * test_percent)

    # Split the files into train, validation, and test sets
    train_files = files[:num_train]
    val_files = files[num_train:num_train + num_val]
    test_files = files[num_train + num_val:]

    # Copy the train files to the output directory
    copy_files_to_output_dir(train_files, dataset_dir, output_dir, "train")

    # Copy the validation files to the output directory
    copy_files_to_output_dir(val_files, dataset_dir, output_dir, "val")

    # Copy the test files to the output directory
    copy_files_to_output_dir(test_files, dataset_dir, output_dir, "test")


if __name__ == '__main__':
    input_dataset_dir = '../data/fake/normal1/'
    output_dataset_dir = '../data/fake/normal'

    split_dataset(input_dataset_dir, output_dataset_dir, 0.8475, 0.1, 0.05)
