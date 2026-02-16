import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import os

class SoilDataset(Dataset):
    """
    Мультимодальный датасет для анализа почвы
    """
    def __init__(self, tabular_data, image_features=None, targets=None):
        self.tabular_data = torch.FloatTensor(tabular_data) if tabular_data is not None else None
        self.image_features = torch.FloatTensor(image_features) if image_features is not None else None
        self.targets = torch.FloatTensor(targets) if targets is not None else None
    
    def __len__(self):
        return len(self.targets) if self.targets is not None else len(self.tabular_data)
    
    def __getitem__(self, idx):
        item = {}
        if self.tabular_data is not None:
            item['tabular'] = self.tabular_data[idx]
        if self.image_features is not None:
            item['image'] = self.image_features[idx]
        if self.targets is not None:
            item['target'] = self.targets[idx]
        return item

def prepare_soil_data(csv_path, target_cols=['pH', 'N', 'P', 'K'], test_size=0.2):
    """
    Подготовка табличных данных о почве
    """
    # Загружаем данные
    df = pd.read_csv(csv_path)
    print(f"Загружено {len(df)} образцов")
    print(f"Колонки: {df.columns.tolist()}")
    
    # Разделяем признаки и цели
    feature_cols = [col for col in df.columns if col not in target_cols]
    X = df[feature_cols].values
    y = df[target_cols].values
    
    # Масштабирование
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y)
    
    # Разделение на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=42
    )
    
    # Создаем датасеты
    train_dataset = SoilDataset(tabular_data=X_train, targets=y_train)
    val_dataset = SoilDataset(tabular_data=X_val, targets=y_val)
    
    return train_dataset, val_dataset, scaler_X, scaler_y, feature_cols, target_cols

def analyze_data_distribution(df, target_cols):
    """
    Анализ распределения данных
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(target_cols):
        axes[i].hist(df[col], bins=30, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
        axes[i].axvline(df[col].mean(), color='red', linestyle='--', label=f'Mean: {df[col].mean():.2f}')
        axes[i].axvline(df[col].median(), color='green', linestyle='--', label=f'Median: {df[col].median():.2f}')
        axes[i].legend()
    
    plt.tight_layout()
    plt.savefig('reports/data_distribution.png')
    plt.show()

def correlation_analysis(df, target_cols):
    """
    Анализ корреляций
    """
    # Корреляционная матрица
    corr_matrix = df.corr()
    
    plt.figure(figsize=(14, 12))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', center=0, 
                square=True, linewidths=0.5)
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig('reports/correlation_matrix.png')
    plt.show()
    
    # Корреляции с целевыми переменными
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.ravel()
    
    for i, target in enumerate(target_cols):
        correlations = corr_matrix[target].drop(target).sort_values(ascending=False)
        top_features = correlations.head(10)
        
        axes[i].barh(range(len(top_features)), top_features.values)
        axes[i].set_yticks(range(len(top_features)))
        axes[i].set_yticklabels(top_features.index)
        axes[i].set_xlabel('Correlation')
        axes[i].set_title(f'Top correlations with {target}')
        axes[i].axvline(x=0, color='black', linestyle='-', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig('reports/target_correlations.png')
    plt.show()

if __name__ == "__main__":
    # Создаем синтетические данные для примера
    np.random.seed(42)
    n_samples = 1000
    
    # Генерируем данные
    data = {
        'depth': np.random.uniform(0, 100, n_samples),
        'sand': np.random.uniform(0, 100, n_samples),
        'clay': np.random.uniform(0, 100, n_samples),
        'silt': np.random.uniform(0, 100, n_samples),
        'organic_matter': np.random.uniform(0, 10, n_samples),
        'cec': np.random.uniform(0, 50, n_samples),
        'ca': np.random.uniform(0, 100, n_samples),
        'mg': np.random.uniform(0, 50, n_samples),
        'na': np.random.uniform(0, 20, n_samples),
        'k': np.random.uniform(0, 30, n_samples),  # Это K в почве
        'pH': np.random.uniform(4, 9, n_samples),
        'N': np.random.uniform(0, 100, n_samples),
        'P': np.random.uniform(0, 50, n_samples),
        'K': np.random.uniform(0, 200, n_samples),  # Это доступный K
    }
    
    df = pd.DataFrame(data)
    
    # Сохраняем
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/soil_data.csv', index=False)
    
    # Анализируем
    analyze_data_distribution(df, ['pH', 'N', 'P', 'K'])
    correlation_analysis(df, ['pH', 'N', 'P', 'K'])
    
    # Готовим датасеты
    train_dataset, val_dataset, scaler_X, scaler_y, features, targets = prepare_soil_data(
        'data/raw/soil_data.csv'
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Features: {features[:5]}...")
    print(f"Targets: {targets}")