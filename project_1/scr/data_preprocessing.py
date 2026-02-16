import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import shutil
from pathlib import Path

class PlantDiseaseDataset(Dataset):
    """
    Датасет для классификации болезней растений
    """
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.classes = []
        
        # Получаем все классы (папки)
        self.classes = sorted([d for d in os.listdir(root_dir) 
                               if os.path.isdir(os.path.join(root_dir, d))])
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Собираем все изображения
        for cls in self.classes:
            cls_path = os.path.join(root_dir, cls)
            for img_name in os.listdir(cls_path):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(cls_path, img_name))
                    self.labels.append(self.class_to_idx[cls])
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_transforms():
    """
    Определяем аугментации для train и валидации
    """
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform

def prepare_data(data_path, batch_size=32, val_split=0.2):
    """
    Подготовка DataLoader'ов для обучения
    """
    train_transform, val_transform = get_transforms()
    
    # Создаем датасет
    full_dataset = PlantDiseaseDataset(data_path, transform=train_transform)
    
    # Разделяем на train и val
    train_size = int((1 - val_split) * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        full_dataset, [train_size, val_size]
    )
    
    # Для валидации используем другой transform
    val_dataset.dataset.transform = val_transform
    
    # Создаем загрузчики
    train_loader = DataLoader(train_dataset, batch_size=batch_size, 
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, 
                           shuffle=False, num_workers=4)
    
    return train_loader, val_loader, full_dataset.classes

def visualize_augmentations(dataset, num_samples=5):
    """
    Визуализация аугментаций для проверки
    """
    fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
    
    for i in range(num_samples):
        # Оригинал
        img, label = dataset[i]
        img = img.numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Original - {dataset.classes[label]}')
        axes[0, i].axis('off')
        
        # С аугментацией
        transform,_ = get_transforms()
        img_aug, _ = dataset[i]
        img_aug = img_aug.numpy().transpose((1, 2, 0))
        img_aug = std * img_aug + mean
        img_aug = np.clip(img_aug, 0, 1)
        
        axes[1, i].imshow(img_aug)
        axes[1, i].set_title('Augmented')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig('reports/augmentation_examples.png')
    plt.show()

if __name__ == "__main__":
    # Пример использования
    train_loader, val_loader, classes = prepare_data('data/raw/PlantVillage')
    print(f"Найдено классов: {len(classes)}")
    print(f"Классы: {classes}")
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")