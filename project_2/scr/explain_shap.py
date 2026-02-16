import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.inspection import permutation_importance
import os
import json

from train_fusion import MLPRegressor, TabularTransformer
from data_preprocessing import prepare_soil_data

class ModelWrapper:
    """
    Обертка для модели в формате, понятном SHAP
    """
    def __init__(self, model, device):
        self.model = model
        self.device = device
    
    def __call__(self, x):
        self.model.eval()
        with torch.no_grad():
            x_tensor = torch.FloatTensor(x).to(self.device)
            outputs = self.model(x_tensor)
        return outputs.cpu().numpy()

def calculate_shap_values(model, X_train, X_val, feature_names, target_names):
    """
    Расчет SHAP значений для интерпретации модели
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_wrapper = ModelWrapper(model, device)
    
    # Используем небольшой background dataset
    background = X_train[:100]
    
    # Создаем объяснитель
    explainer = shap.KernelExplainer(model_wrapper, background)
    
    # Рассчитываем SHAP значения для валидационной выборки
    shap_values = explainer.shap_values(X_val[:50], nsamples=100)
    
    return explainer, shap_values

def plot_shap_summary(shap_values, X_val, feature_names, target_names, target_idx=0):
    """
    Визуализация SHAP summary plot
    """
    plt.figure(figsize=(12, 8))
    
    # Для конкретной целевой переменной
    shap.summary_plot(
        shap_values[target_idx], 
        X_val[:50], 
        feature_names=feature_names,
        show=False,
        max_display=15
    )
    plt.title(f'SHAP Feature Importance for {target_names[target_idx]}')
    plt.tight_layout()
    plt.savefig(f'reports/shap_summary_{target_names[target_idx]}.png', bbox_inches='tight')
    plt.show()

def plot_shap_dependence(shap_values, X_val, feature_names, target_names, feature_idx, target_idx=0):
    """
    Визуализация зависимости SHAP значения от значения признака
    """
    plt.figure(figsize=(10, 6))
    
    shap.dependence_plot(
        feature_idx, 
        shap_values[target_idx], 
        X_val[:50], 
        feature_names=feature_names,
        show=False
    )
    plt.title(f'SHAP Dependence Plot: {feature_names[feature_idx]} for {target_names[target_idx]}')
    plt.tight_layout()
    plt.savefig(f'reports/shap_dependence_{feature_names[feature_idx]}_{target_names[target_idx]}.png', 
                bbox_inches='tight')
    plt.show()

def calculate_permutation_importance(model, X_val, y_val, feature_names, target_names):
    """
    Расчет важности признаков методом перестановок
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    def score_fn(X, y):
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            pred = model(X_tensor).cpu().numpy()
        return -np.mean((y - pred) ** 2)  # отрицательный MSE (чем больше, тем лучше)
    
    # Рассчитываем важность для каждой целевой переменной
    importance_results = {}
    
    for i, target in enumerate(target_names):
        result = permutation_importance(
            lambda X: score_fn(X, y_val[:, i:i+1]),
            X_val, y_val[:, i],
            n_repeats=10,
            random_state=42,
            n_jobs=-1
        )
        
        importance_results[target] = {
            'importances_mean': result.importances_mean,
            'importances_std': result.importances_std
        }
    
    return importance_results

def plot_feature_importance(importance_results, feature_names, target_names):
    """
    Визуализация важности признаков
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.ravel()
    
    for i, target in enumerate(target_names):
        means = importance_results[target]['importances_mean']
        stds = importance_results[target]['importances_std']
        
        # Сортируем по важности
        sorted_idx = np.argsort(means)[::-1]
        top_n = 10
        
        axes[i].barh(range(top_n), means[sorted_idx][:top_n][::-1], 
                    xerr=stds[sorted_idx][:top_n][::-1])
        axes[i].set_yticks(range(top_n))
        axes[i].set_yticklabels([feature_names[j] for j in sorted_idx[:top_n]][::-1])
        axes[i].set_xlabel('Importance')
        axes[i].set_title(f'Feature Importance for {target}')
        axes[i].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('reports/feature_importance.png')
    plt.show()

def main():
    # Загружаем данные
    train_dataset, val_dataset, scaler_X, scaler_y, features, targets = prepare_soil_data(
        'data/raw/soil_data.csv'
    )
    
    # Загружаем лучшую модель
    input_dim = train_dataset.tabular_data.shape[1]
    model = MLPRegressor(input_dim)  # или загрузить сохраненную модель
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.load_state_dict(torch.load('models/best_mlp.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # Подготавливаем данные для анализа
    X_train = train_dataset.tabular_data.numpy()
    X_val = val_dataset.tabular_data.numpy()
    y_val = val_dataset.targets.numpy()
    
    # SHAP анализ
    print("Расчет SHAP значений...")
    explainer, shap_values = calculate_shap_values(model, X_train, X_val, features, targets)
    
    # Визуализация SHAP
    for i, target in enumerate(targets):
        plot_shap_summary(shap_values, X_val, features, targets, target_idx=i)
    
    # Важность признаков методом перестановок
    print("Расчет важности признаков методом перестановок...")
    importance_results = calculate_permutation_importance(model, X_val, y_val, features, targets)
    plot_feature_importance(importance_results, features, targets)
    
    # Сохраняем результаты
    results_summary = {}
    for target in targets:
        results_summary[target] = {
            'top_features': [
                features[i] for i in np.argsort(importance_results[target]['importances_mean'])[-5:][::-1]
            ]
        }
    
    with open('reports/feature_importance_summary.json', 'w') as f:
        json.dump(results_summary, f, indent=4)
    
    print("\n✅ Анализ завершен! Результаты сохранены в 'reports/'")

if __name__ == "__main__":
    main()