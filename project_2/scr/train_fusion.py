import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import os
import json

from data_preprocessing import prepare_soil_data

class TabularTransformer(nn.Module):
    """
    FT-Transformer для табличных данных
    """
    def __init__(self, input_dim, hidden_dim=256, num_heads=8, num_layers=3, dropout=0.1):
        super().__init__()
        
        # Feature Tokenizer
        self.feature_tokenizer = nn.Linear(1, hidden_dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output head
        self.output_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 4)  # 4 target variables
        )
    
    def forward(self, x):
        # x shape: (batch_size, n_features)
        batch_size, n_features = x.shape
        
        # Tokenize each feature
        x = x.unsqueeze(-1)  # (batch_size, n_features, 1)
        tokens = self.feature_tokenizer(x)  # (batch_size, n_features, hidden_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        tokens = torch.cat([cls_tokens, tokens], dim=1)  # (batch_size, n_features+1, hidden_dim)
        
        # Apply transformer
        transformed = self.transformer(tokens)
        
        # Use CLS token for prediction
        cls_output = transformed[:, 0, :]
        
        return self.output_head(cls_output)

class MLPRegressor(nn.Module):
    """
    Простой MLP для регрессии
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
        
        layers.append(nn.Linear(prev_dim, 4))  # 4 target variables
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)

class FusionModel(nn.Module):
    """
    Гибридная модель для изображений и табличных данных
    (упрощенная версия - только табличные данные)
    """
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        
        self.tabular_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU()
        )
        
        # В реальном проекте здесь был бы CNN для изображений
        
        self.regressor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Linear(64, 4)  # 4 target variables
        )
    
    def forward(self, x):
        tab_features = self.tabular_encoder(x)
        return self.regressor(tab_features)

class SoilTrainer:
    def __init__(self, model, train_dataset, val_dataset, model_name='fusion'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_name = model_name
        
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', patience=10, factor=0.5
        )
        
        self.train_losses = []
        self.val_losses = []
        self.val_metrics = []
    
    def train_epoch(self, train_loader):
        self.model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc='Training')
        for batch in pbar:
            x = batch['tabular'].to(self.device)
            y = batch['target'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(x)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'loss': running_loss / len(train_loader)})
        
        return running_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        running_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for batch in pbar:
                x = batch['tabular'].to(self.device)
                y = batch['target'].to(self.device)
                
                outputs = self.model(x)
                loss = self.criterion(outputs, y)
                
                running_loss += loss.item()
                all_preds.append(outputs.cpu().numpy())
                all_targets.append(y.cpu().numpy())
        
        all_preds = np.vstack(all_preds)
        all_targets = np.vstack(all_targets)
        
        # Calculate metrics
        mse = mean_squared_error(all_targets, all_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(all_targets, all_preds)
        r2 = r2_score(all_targets, all_preds, multioutput='variance_weighted')
        
        metrics = {
            'loss': running_loss / len(val_loader),
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        return metrics, all_preds, all_targets
    
    def train(self, batch_size=32, epochs=100):
        train_loader = DataLoader(self.train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(self.val_dataset, batch_size=batch_size)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(epochs):
            print(f"\nEpoch {epoch+1}/{epochs}")
            print("-" * 50)
            
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)
            
            val_metrics, _, _ = self.validate(val_loader)
            self.val_losses.append(val_metrics['loss'])
            self.val_metrics.append(val_metrics)
            
            self.scheduler.step(val_metrics['loss'])
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}")
            print(f"Val RMSE: {val_metrics['rmse']:.4f}")
            print(f"Val R²: {val_metrics['r2']:.4f}")
            
                       if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                best_model_state = self.model.state_dict().copy()
                torch.save(best_model_state, f'models/best_{self.model_name}.pth')
                print(f"✓ Сохранена лучшая модель")
        
        self.plot_training_history()
        return best_model_state
    
    def plot_training_history(self):
        """
        Визуализация процесса обучения
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # График потерь
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # RMSE
        rmse_history = [m['rmse'] for m in self.val_metrics]
        axes[1].plot(rmse_history, color='orange')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('RMSE')
        axes[1].set_title('Validation RMSE')
        axes[1].grid(True)
        
        # R²
        r2_history = [m['r2'] for m in self.val_metrics]
        axes[2].plot(r2_history, color='green')
        axes[2].set_xlabel('Epoch')
        axes[2].set_ylabel('R²')
        axes[2].set_title('Validation R²')
        axes[2].grid(True)
        
        plt.tight_layout()
        plt.savefig(f'reports/training_history_{self.model_name}.png')
        plt.show()

def compare_models(train_dataset, val_dataset):
    """
    Сравнение разных моделей
    """
    input_dim = train_dataset.tabular_data.shape[1]
    
    models = {
        'mlp': MLPRegressor(input_dim),
        'transformer': TabularTransformer(input_dim),
        'fusion': FusionModel(input_dim)
    }
    
    results = {}
    
    for name, model in models.items():
        print(f"\n{'='*60}")
        print(f"Обучение модели: {name}")
        print('='*60)
        
        trainer = SoilTrainer(model, train_dataset, val_dataset, model_name=name)
        trainer.train(epochs=50)
        
        # Финальная валидация
        val_loader = DataLoader(val_dataset, batch_size=32)
        metrics, preds, targets = trainer.validate(val_loader)
        results[name] = metrics
        
        print(f"\nФинальные метрики для {name}:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  R²: {metrics['r2']:.4f}")
    
    # Сравнительный график
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    models_list = list(results.keys())
    rmse_values = [results[m]['rmse'] for m in models_list]
    mae_values = [results[m]['mae'] for m in models_list]
    r2_values = [results[m]['r2'] for m in models_list]
    
    axes[0].bar(models_list, rmse_values)
    axes[0].set_title('RMSE Comparison')
    axes[0].set_ylabel('RMSE')
    
    axes[1].bar(models_list, mae_values)
    axes[1].set_title('MAE Comparison')
    axes[1].set_ylabel('MAE')
    
    axes[2].bar(models_list, r2_values)
    axes[2].set_title('R² Comparison')
    axes[2].set_ylabel('R²')
    
    plt.tight_layout()
    plt.savefig('reports/model_comparison.png')
    plt.show()
    
    return results

def main():
    # Создаем папки
    os.makedirs('models', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Загружаем данные
    train_dataset, val_dataset, scaler_X, scaler_y, features, targets = prepare_soil_data(
        'data/raw/soil_data.csv'
    )
    
    # Сравниваем модели
    results = compare_models(train_dataset, val_dataset)
    
    # Сохраняем результаты
    with open('reports/comparison_results.json', 'w') as f:
        # Преобразуем numpy значения в обычные float
        results_serializable = {}
        for model, metrics in results.items():
            results_serializable[model] = {k: float(v) for k, v in metrics.items()}
        json.dump(results_serializable, f, indent=4)
    
    print("\n✅ Обучение завершено! Результаты сохранены в 'reports/'")

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    main()