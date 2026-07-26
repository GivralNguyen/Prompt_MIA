import torchvision.transforms as transforms

def build_data_transform(normalize_stats, resize, centercrop_size,
                         interpolation_type=transforms.InterpolationMode.BILINEAR,
                         train_augment=False):
    if train_augment:
        transform = transforms.Compose([
            transforms.Resize(resize, interpolation=interpolation_type),
            transforms.RandomCrop(size=centercrop_size, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(*normalize_stats, inplace=True)
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(resize, interpolation=interpolation_type),
            transforms.CenterCrop(size=centercrop_size),
                                    transforms.ToTensor(),
                                    transforms.Normalize(*normalize_stats, inplace=True)
        ])
    return transform