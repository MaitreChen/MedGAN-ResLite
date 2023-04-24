import torchvision.transforms as transforms
import torch.utils.data as data

from medmnist import INFO
import medmnist


def get_data_loader(data_flag, BATCH_SIZE, workers=8, download=True):
    info = INFO[data_flag]

    DataClass = getattr(medmnist, info['python_class'])

    # preprocessing
    data_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[.5], std=[.5])
    ])

    # load the data
    train_dataset = DataClass(split='train', transform=data_transform, download=download)
    val_dataset = DataClass(split='val', transform=data_transform, download=download)
    test_dataset = DataClass(split='test', transform=data_transform, download=download)

    print(f"\ndataset: {data_flag}")
    print(f"train dataset: {len(train_dataset)}")
    print(f"val dataset: {len(val_dataset)}")
    print(f"test dataset: {len(test_dataset)}\n")

    # encapsulate data into dataloader form
    train_loader = data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=workers)
    val_loader = data.DataLoader(dataset=val_dataset, batch_size=2 * BATCH_SIZE, shuffle=False, num_workers=workers)
    test_loader = data.DataLoader(dataset=test_dataset, batch_size=2 * BATCH_SIZE, shuffle=False, num_workers=workers)

    return train_loader, val_loader, test_loader


if __name__ == '__main__':
    data_flag = 'real'
    BATCH_SIZE = 32

    train_dataloader, val_dataloader, test_dataloader = get_data_loader(data_flag, BATCH_SIZE)
    normal = 0
    abnormal = 0

    labels = train_dataloader.dataset.labels
    num_of_images = len(labels)
    for i in range(num_of_images):
        if labels[i][0] == 0:
            normal += 1
        else:
            abnormal += 1
    print(f"normal={normal}")
    print(f"pneumonia={abnormal}")
