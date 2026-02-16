import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import os

from train import PlantDiseaseClassifier
from data_preprocessing import prepare_data

def plot_confusion_matrix(cm, classes, save_path='reports/confusion_matrix.png'):
    """
    Визуализация матрицы ошибок
    """
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()

def calculate_metrics(all_preds, all_labels, classes):
    """
    Расчет всех метрик
    """
    # Classification report
    report = classification_report(all_labels, all_preds, 
                                 target_names=classes, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class metrics
    per_class_acc = cm.diagonal() / cm.sum(axis=1)
    
    # Macro F1
    macro_f1 = f1_score(all_labels, all_preds, average='macro')
    
    return report, cm, per_class_acc, macro_f1

def evaluate_model(model_path, data_path, classes):
    """
    Полная оценка модели
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загружаем модель
    num_classes = len(classes)
    classifier = PlantDiseaseClassifier(num_classes, model_name='efficientnet')
    classifier.model.load_state_dict(torch.load(model_path, map_location=device))
    classifier.model.eval()
    
    # Загружаем тестовые данные
    _, test_loader, _ = prepare_data(data_path, batch_size=32)
    
    # Получаем предсказания
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = classifier.model(images)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Рассчитываем метрики
    report, cm, per_class_acc, macro_f1 = calculate_metrics(all_preds, all_labels, classes)
    
    # Выводим результаты
    print("=" * 60)
    print("РЕЗУЛЬТАТЫ ОЦЕНКИ МОДЕЛИ")
    print("=" * 60)
    
    print(f"\nMacro F1-Score: {macro_f1:.4f}")
    
    print("\nТочность по классам:")
    for i, cls in enumerate(classes):
        print(f"  {cls}: {per_class_acc[i]:.2%}")
    
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Визуализируем матрицу ошибок
    plot_confusion_matrix(cm, classes)
    
    # Сохраняем отчет
    with open('reports/evaluation_report.txt', 'w') as f:
        f.write("CLASSIFICATION REPORT\n")
        f.write("=" * 50 + "\n")
        for key, value in report.items():
            f.write(f"{key}: {value}\n")
        f.write(f"\nMacro F1-Score: {macro_f1:.4f}\n")
    
    return report, cm, macro_f1

def visualize_predictions(model_path, data_path, classes, num_samples=10):
    """
    Визуализация предсказаний на случайных примерах
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загружаем модель
    num_classes = len(classes)
    classifier = PlantDiseaseClassifier(num_classes, model_name='efficientnet')
    classifier.model.load_state_dict(torch.load(model_path, map_location=device))
    classifier.model.eval()
    
    # Загружаем данные
    _, test_loader, _ = prepare_data(data_path, batch_size=32)
    
    # Берем несколько случайных батчей
    images, labels = next(iter(test_loader))
    images, labels = images[:num_samples], labels[:num_samples]
    
    # Получаем предсказания
    with torch.no_grad():
        outputs = classifier.model(images.to(device))
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = outputs.max(1)
    
    # Визуализация
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    axes = axes.ravel()
    
    for i in range(num_samples):
        # Преобразуем изображение для отображения
        img = images[i].cpu().numpy().transpose((1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        img = np.clip(img, 0, 1)
        
        axes[i].imshow(img)
        
        true_class = classes[labels[i]]
        pred_class = classes[predicted[i]]
        prob = probabilities[i][predicted[i]].item()
        
        color = 'green' if labels[i] == predicted[i] else 'red'
        axes[i].set_title(f'True: {true_class}\nPred: {pred_class}\nProb: {prob:.2%}', 
                         color=color)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('reports/sample_predictions.png')
    plt.show()

if __name__ == "__main__":
    # Параметры
    data_path = 'data/raw/PlantVillage'
    model_path = 'models/best_model.pth'
    
    # Получаем классы
    _, _, classes = prepare_data(data_path, batch_size=32)
    
    # Оценка модели
    report, cm, macro_f1 = evaluate_model(model_path, data_path, classes)
    
    # Визуализация предсказаний
    visualize_predictions(model_path, data_path, classes)
    
    print("\n✅ Оценка завершена! Результаты сохранены в папке 'reports/'")