from medmnist import INFO
import medmnist

import cv2 as cv
import os


def save(data_flag, download=False):
    info = INFO[data_flag]

    DataClass = getattr(medmnist, info['python_class'])

    # load the data
    train_dataset = DataClass(split='train', download=download)
    val_dataset = DataClass(split='val', download=download)
    test_dataset = DataClass(split='test', download=download)

    print(f"dataset: {data_flag}")
    print(f"train dataset: {len(train_dataset)}")
    print(f"val dataset: {len(val_dataset)}")
    print(f"test dataset: {len(test_dataset)}\n")

    print(f"The shape of image: {train_dataset.imgs[1].shape}")
    print(f"The type of image: {type(train_dataset.imgs[1])}")

    # create dir and different dataset dir
    dataset_path = {'train': os.path.join(save_dir, 'train'),
                    'val': os.path.join(save_dir, 'val'),
                    'test': os.path.join(save_dir, 'test')
                    }

    if not os.path.exists(save_dir):
        os.makedirs(os.path.join(save_dir, 'train'))
        os.makedirs(os.path.join(save_dir, 'test'))
        os.makedirs(os.path.join(save_dir, 'val'))

        for i, v in dataset_path.items():
            os.makedirs(os.path.join(v, 'normal'))  # 0
            os.makedirs(os.path.join(v, 'pneumonia'))  # 1

    data_info = {'train': train_dataset, 'val': val_dataset, 'test': test_dataset}
    mode_set = ['train', 'val', 'test']
    label_map = {
        0: "normal",
        1: "pneumonia"
    }

    normal_num = 0
    pneumonia_num = 0
    for mode in mode_set:
        for i in range(len(data_info[mode])):

            # attention
            # img = cv.resize(data_info[mode].imgs[i], (64, 64), interpolation=cv.INTER_LINEAR)
            img = data_info[mode].imgs[i]
            label = data_info[mode].labels[i].item()

            if label == 1:
                pneumonia_num += 1
                cv.imwrite(f'{dataset_path[mode]}/{label_map[label]}/img_{pneumonia_num}.png', img)
            else:
                normal_num += 1
                cv.imwrite(f'{dataset_path[mode]}/{label_map[label]}/img_{normal_num}.png', img)

    print(f"normal num: {normal_num}")
    print(f"pneumonia num: {pneumonia_num}")


if __name__ == '__main__':
    data_name = 'pneumonia'

    # data_path = f"./data/{data_name}mnist_reserve/"
    data_path = f"../data/{data_name}mnist/"

    data_flag = data_name + 'mnist'

    save_dir = f'../data/{data_flag}'

    # save
    save(data_flag, True)
