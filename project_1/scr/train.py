import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import os
from datetime import datetime

from data_preprocessing import prepare_data

class PlantDiseaseClassifier:
    def __init__(self, num_classes, model_name='efficientnet', device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = num_classes
        self.model_name = model_name
        self.model = self._build_model()
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
        print(f"Используется устройство: {self.device}")
        print(f"Модель: {model_name}")
        print(f"Количество классов: {num_classes}")
    
    def _build_model(self):
        """
        Создание модели на основе выбранной архитектуры
        """
        if 'resnet' in self.model_name.lower():
            model = models.resnet50(pretrained=True)
            num_features = model.fc.in_features
            model.fc = nn.Linear(num_features, self.num_classes)
            
        elif 'efficientnet' in self.model_name.lower():
            model = models.efficientnet_b3(pretrained=True)
            num_features = model.classifier[1].in_features
            model.classifier[1] = nn.Linear(num_features, self.num_classes)
            
        elif 'vit' in self.model_name.lower():
            model = models.vit_b_16(pretrained=True)
            num_features = model.heads.head.in_features
            model.heads.head = nn.Linear(num_features, self.num_classes)
            
        else:
            raise ValueError(f"Неизвестная модель: {self.model_name}")
        
        return model.to(self.device)
    
    def train_epoch(self, train_loader):
        """
        Одна эпоха обучения
        """
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
        for images, labels in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
        return running_loss/len(train_loader), 100.*correct/total
    
    def validate(self, val_loader):
        """
        Валидация модели
        """
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for images, labels in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': running_loss/total, 'acc': 100.*correct/total})
        
        return running_loss/len(val_loader), 100.*correct/total, all_preds, all_labels
    
    def train(self, train_loader, val_loader, epochs=30, lr=0.001):
        """
        Полный цикл обучения
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=5, factor=0.1
        )
        
        best_val_acc = 0.0
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            # Обучение
            train_loss, train_acc = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)
            
            # Валидация
            val_loss, val_acc, _, _ = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_accs.append(val_acc)
            
            # Снижение learning rate
            self.scheduler.step(val_loss)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Сохраняем лучшую модель
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_model_state = self.model.state_dict().copy()
                torch.save(best_model_state, 'models/best_model.pth')
                print(f"✓ Сохранена лучшая модель с val_acc = {val_acc:.2f}%")
        
        self.plot_training_history()
        return best_model_state
    
    def plot_training_history(self):
        """
        Визуализация процесса обучения
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # График потерь
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.legend()
        ax1.grid(True)
        
        # График точности
        ax2.plot(self.train_accs, label='Train Acc')
        ax2.plot(self.val_accs, label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.set_title('Training and Validation Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig('reports/training_history.png')
        plt.show()

def main():
    # Параметры
    data_path = 'data/raw/PlantVillage'
    batch_size = 32
    epochs = 30
    lr = 0.001
    model_name = 'efficientnet'  # или 'resnet', 'vit'
    
    # Создаем папки для сохранения
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Загружаем данные
    print("Загрузка данных...")
    train_loader, val_loader, classes = prepare_data(data_path, batch_size)
    num_classes = len(classes)
    
    # Создаем и обучаем модель
    classifier = PlantDiseaseClassifier(num_classes, model_name)
    classifier.train(train_loader, val_loader, epochs, lr)
    
    print("\n✅ Обучение завершено!")
    print(f"Лучшая модель сохранена в 'models/best_model.pth'")

if __name__ == "__main__":
    main()