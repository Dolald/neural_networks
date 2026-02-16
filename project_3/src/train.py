import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
import json

class FertilizerDataset(Dataset):
    """
    Датасет для рекомендаций удобрений
    """
    def __init__(self, features, targets):
        self.features = torch.FloatTensor(features)
        self.targets = torch.FloatTensor(targets)
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'targets': self.targets[idx]
        }

class MultiOutputMLP(nn.Module):
    """
    Многослойный перцептрон для множественной регрессии
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        # Выходной слой для N, P, K рекомендаций
        self.shared_layers = nn.Sequential(*layers)
        
        # Отдельные головы для каждого выходного параметра
        self.N_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.P_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
        
        self.K_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        shared = self.shared_layers(x)
        N_out = self.N_head(shared)
        P_out = self.P_head(shared)
        K_out = self.K_head(shared)
        
        return torch.cat([N_out, P_out, K_out], dim=1)

class YieldSimulator:
    """
    Симулятор урожайности для оценки рекомендаций
    """
    def __init__(self, soil_data, weather_data):
        self.soil_data = soil_data
        self.weather_data = weather_data
    
    def simulate_yield(self, N, P, K, soil_idx):
        """
        Симуляция урожайности на основе доз удобрений
        """
        soil = self.soil_data[soil_idx]
        weather = self.weather_data[soil_idx % len(self.weather_data)]
        
        # Базовая урожайность
        base_yield = (
            3.0 +
            0.1 * (soil['pH'] - 6.5) +
            0.02 * soil['organic_matter'] +
            0.05 * (weather['temperature'] - 20)
        )
        
        # Эффект удобрений (закон убывающей отдачи)
        N_effect = 0.03 * N - 0.0001 * N**2
        P_effect = 0.02 * P - 0.0001 * P**2
        K_effect = 0.015 * K - 0.00005 * K**2
        
        total_yield = base_yield + N_effect + P_effect + K_effect
        return max(0, total_yield)

class FertilizerTrainer:
    def __init__(self, model, train_dataset, val_dataset):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            x = batch['features'].to(self.device)
            y = batch['targets'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x)
            
            # Потери для каждого выходного параметра
            loss_N = self.criterion(outputs[:, 0:1], y[:, 0:1])
            loss_P = self.criterion(outputs[:, 1:2], y[:, 1:2])
            loss_K = self.criterion(outputs[:, 2:3], y[:, 2:3])
            loss = loss_N + loss_P + loss_K
            
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            all_preds.append(outputs.detach().cpu().numpy())
            all_targets.append(y.cpu().numpy())
            
            pbar.set_postfix({'loss': running_loss / len(train_loader)})
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        metrics = self.calculate_metrics(all_preds, all_targets)
        
        return running_loss / len(train_loader), metrics
    
    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch in pbar:
                x = batch['features'].to(self.device)
                y = batch['targets'].to(self.device)
                
                outputs = self.model(x)
                
                loss_N = self.criterion(outputs[:, 0:1], y[:, 0:1])
                loss_P = self.criterion(outputs[:, 1:2], y[:, 1:2])
                loss_K = self.criterion(outputs[:, 2:3], y[:, 2:3])
                loss = loss_N + loss_P + loss_K
                
                running_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        metrics = self.calculate_metrics(all_preds, all_targets)
        
        return running_loss / len(val_loader), metrics, all_preds, all_targets
    
    def calculate_metrics(self, preds, targets):
        """
        Расчет метрик для каждой выходной переменной
        """
        metrics = {}
        names = ['N', 'P', 'K']
        
        for i, name in enumerate(names):
            mae = np.mean(np.abs(preds[:, i] - targets[:, i]))
            mse = np.mean((preds[:, i] - targets[:, i])**2)
            rmse = np.sqrt(mse)
            
            metrics[name] = {
                'mae': mae,
                'rmse': rmse,
                'mse': mse
            }
        
        # Общие метрики
        metrics['combined'] = {
            'mae': np.mean([metrics[n]['mae'] for n in names]),
            'rmse': np.sqrt(np.mean([metrics[n]['mse'] for n in names]))
        }
        
        return metrics
    
    def train(self, batch_size=32, epochs=100):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            train_loss, train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            self.train_metrics.append(train_metrics)
            
            val_loss, val_metrics, _, _ = self.validate(val_loader)
            self.val_losses.append(val_loss)
            self.val_metrics.append(val_metrics)
            
            self.scheduler.step(val_loss)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_loss:.4f}")
            print(f"Val MAE - N: {val_metrics['N']['mae']:.2f}, "
                  f"P: {val_metrics['P']['mae']:.2f}, "
                  f"K: {val_metrics['K']['mae']:.2f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                torch.save(best_model_state, 'models/best_model.pth')
                print(f"✓ Сохранена лучшая модель")
        
        self.plot_training_history()
        return best_model_state
    
    def plot_training_history(self):
        """
        Визуализация процесса обучения
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # График потерь
        axes[0, 0].plot(self.train_losses, label='Train Loss')
        axes[0, 0].plot(self.val_losses, label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # MAE для каждого элемента
        for i, nutrient in enumerate(['N', 'P', 'K']):
            train_mae = [m[nutrient]['mae'] for m in self.train_metrics]
            val_mae = [m[nutrient]['mae'] for m in self.val_metrics]
            
            axes[0, 1].plot(train_mae, linestyle='--', label=f'Train {nutrient} MAE')
            axes[0, 1].plot(val_mae, label=f'Val {nutrient} MAE')
        
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('MAE')
        axes[0, 1].set_title('MAE by Nutrient')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # RMSE
        train_rmse = [m['combined']['rmse'] for m in self.train_metrics]
        val_rmse = [m['combined']['rmse'] for m in self.val_metrics]
        
        axes[1, 0].plot(train_rmse, label='Train RMSE')
        axes[1, 0].plot(val_rmse, label='Val RMSE')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('RMSE')
        axes[1, 0].set_title('Combined RMSE')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # Learning rate
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig('reports/training_history.png')
        plt.show()

def prepare_data(features_path, targets_path, test_size=0.2):
    """
    Подготовка данных для обучения
    """
    # Загружаем данные
    features = pd.read_csv(features_path)
    targets = pd.read_csv(targets_path)[['N_recommendation', 'P_recommendation', 'K_recommendation']]
    
    # Масштабируем признаки
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    
    X_scaled = scaler_X.fit_transform(features)
    y_scaled = scaler_y.fit_transform(targets)
    
    # Разделяем на train/val
    X_train, X_val, y_train, y_val = train_test_split(
        X_scaled, y_scaled, test_size=test_size, random_state=42
    )
    
    # Создаем датасеты
    train_dataset = FertilizerDataset(X_train, y_train)
    val_dataset = FertilizerDataset(X_val, y_val)
    
    return train_dataset, val_dataset, scaler_X, scaler_y, features.columns

def main():
    # Создаем папки
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Загружаем данные
    train_dataset, val_dataset, scaler_X, scaler_y, feature_names = prepare_data(
        'data/raw/features.csv',
        'data/raw/targets.csv'
    )
    
    # Создаем модель
    input_dim = train_dataset.features.shape[1]
    model = MultiOutputMLP(input_dim)
    
    # Обучаем
    trainer = FertilizerTrainer(model, train_dataset, val_dataset)
    trainer.train(epochs=50)
    
    # Финальная оценка
    val_loader = DataLoader(val_dataset, batch_size=32)
    val_loss, val_metrics, preds, targets = trainer.validate(val_loader)
    
    print("\n" + "="*60)
    print("ФИНАЛЬНЫЕ РЕЗУЛЬТАТЫ")
    print("="*60)
    
    for nutrient in ['N', 'P', 'K']:
        print(f"{nutrient} - MAE: {val_metrics[nutrient]['mae']:.2f}, "
              f"RMSE: {val_metrics[nutrient]['rmse']:.2f}")
    
    # Сохраняем результаты
    results = {
        'final_metrics': {
            k: {kk: float(vv) for kk, vv in v.items()} 
            for k, v in val_metrics.items()
        }
    }
    
    with open('reports/final_results.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n✅ Обучение завершено!")

if __name__ == "__main__":
    main()