import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
import os
import json

from train import MultiOutputMLP, FertilizerDataset, YieldSimulator, prepare_data
from torch.utils.data import DataLoader

def evaluate_predictions(model, val_dataset, scaler_y, device):
    """
    Оценка точности предсказаний доз удобрений
    """
    val_loader = DataLoader(val_dataset, batch_size=32)
    model.eval()
    
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch['features'].to(device)
            y = batch['targets'].to(device)
            
            outputs = model(x)
            all_preds.append(outputs.cpu().numpy())
            all_targets.append(y.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Обратное масштабирование
    all_preds_orig = scaler_y.inverse_transform(all_preds)
    all_targets_orig = scaler_y.inverse_transform(all_targets)
    
    # Расчет метрик
    metrics = {}
    nutrients = ['N', 'P', 'K']
    
    for i, nutrient in enumerate(nutrients):
        mae = mean_absolute_error(all_targets_orig[:, i], all_preds_orig[:, i])
        rmse = np.sqrt(mean_squared_error(all_targets_orig[:, i], all_preds_orig[:, i]))
        mape = np.mean(np.abs((all_targets_orig[:, i] - all_preds_orig[:, i]) / 
                              (all_targets_orig[:, i] + 1e-6))) * 100
        
        metrics[nutrient] = {
            'mae': mae,
            'rmse': rmse,
            'mape': mape
        }
    
    return metrics, all_preds_orig, all_targets_orig

def simulate_yield_comparison(model, val_dataset, scaler_y, features_df, device):
    """
    Сравнение урожайности при разных стратегиях внесения удобрений
    """
    val_loader = DataLoader(val_dataset, batch_size=32)
    model.eval()
    
    # Получаем предсказания модели
    all_preds = []
    all_features = []
    
    with torch.no_grad():
        for batch in val_loader:
            x = batch['features'].to(device)
            y = batch['targets'].to(device)
            
            outputs = model(x)
            all_preds.append(outputs.cpu().numpy())
            all_features.append(x.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_features = np.vstack(all_features)
    
    # Обратное масштабирование
    all_preds_orig = scaler_y.inverse_transform(all_preds)
    
    # Получаем исходные данные для симуляции
    soil_data = []
    weather_data = []
    
    # Упрощенная симуляция
    yields_model = []
    yields_baseline = []
    yields_uniform = []
    
    for i in range(len(all_preds_orig)):
        # Дозы от модели
        N_model, P_model, K_model = all_preds_orig[i]
        
        # Базовые дозы (средние по датасету)
        N_base = np.mean(all_preds_orig[:, 0])
        P_base = np.mean(all_preds_orig[:, 1])
        K_base = np.mean(all_preds_orig[:, 2])
        
        # Единые дозы для всех
        N_uniform = 100
        P_uniform = 50
        K_uniform = 150
        
        # Симуляция урожайности
        # В реальном проекте здесь была бы сложная модель
        # Для демонстрации используем упрощенную формулу
        def calc_yield(N, P, K):
            base = 3.0
            N_eff = 0.03 * N - 0.0001 * N**2
            P_eff = 0.02 * P - 0.0001 * P**2
            K_eff = 0.015 * K - 0.00005 * K**2
            noise = np.random.normal(0, 0.1)
            return base + N_eff + P_eff + K_eff + noise
        
        yields_model.append(calc_yield(N_model, P_model, K_model))
        yields_baseline.append(calc_yield(N_base, P_base, K_base))
        yields_uniform.append(calc_yield(N_uniform, P_uniform, K_uniform))
    
    yields_model = np.array(yields_model)
    yields_baseline = np.array(yields_baseline)
    yields_uniform = np.array(yields_uniform)
    
    # Расчет улучшений
    improvement_vs_baseline = (yields_model - yields_baseline) / yields_baseline * 100
    improvement_vs_uniform = (yields_model - yields_uniform) / yields_uniform * 100
    
    results = {
        'mean_yield_model': np.mean(yields_model),
        'mean_yield_baseline': np.mean(yields_baseline),
        'mean_yield_uniform': np.mean(yields_uniform),
        'improvement_vs_baseline': np.mean(improvement_vs_baseline),
        'improvement_vs_uniform': np.mean(improvement_vs_uniform),
        'std_improvement_vs_baseline': np.std(improvement_vs_baseline),
        'std_improvement_vs_uniform': np.std(improvement_vs_uniform)
    }
    
    return results, yields_model, yields_baseline, yields_uniform

def plot_recommendation_analysis(preds, targets, nutrients, save_dir='reports'):
    """
    Визуализация качества рекомендаций
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    for i, nutrient in enumerate(nutrients):
        # Scatter plot предсказаний vs реальных значений
        axes[0, i].scatter(targets[:, i], preds[:, i], alpha=0.5)
        axes[0, i].plot([targets[:, i].min(), targets[:, i].max()], 
                       [targets[:, i].min(), targets[:, i].max()], 
                       'r--', label='Идеальная линия')
        axes[0, i].set_xlabel(f'Фактическая доза {nutrient}')
        axes[0, i].set_ylabel(f'Предсказанная доза {nutrient}')
        axes[0, i].set_title(f'{nutrient}: Предсказание vs Факт')
        axes[0, i].legend()
        axes[0, i].grid(True, alpha=0.3)
        
        # Распределение ошибок
        errors = preds[:, i] - targets[:, i]
        axes[1, i].hist(errors, bins=50, edgecolor='black', alpha=0.7)
        axes[1, i].axvline(x=0, color='red', linestyle='--', label='Нулевая ошибка')
        axes[1, i].set_xlabel(f'Ошибка предсказания {nutrient}')
        axes[1, i].set_ylabel('Частота')
        axes[1, i].set_title(f'{nutrient}: Распределение ошибок')
        axes[1, i].legend()
        axes[1, i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/recommendation_analysis.png')
    plt.show()

def plot_yield_comparison(results, yields_model, yields_baseline, yields_uniform, save_dir='reports'):
    """
    Визуализация сравнения урожайности
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Box plot урожайности
    data_to_plot = [yields_model, yields_baseline, yields_uniform]
    axes[0].boxplot(data_to_plot, labels=['Модель', 'Базовый', 'Единые дозы'])
    axes[0].set_ylabel('Урожайность (т/га)')
    axes[0].set_title('Сравнение урожайности при разных стратегиях')
    axes[0].grid(True, alpha=0.3)
    
    # Гистограммы
    axes[1].hist(yields_model, bins=30, alpha=0.5, label='Модель', density=True)
    axes[1].hist(yields_baseline, bins=30, alpha=0.5, label='Базовый', density=True)
    axes[1].hist(yields_uniform, bins=30, alpha=0.5, label='Единые дозы', density=True)
    axes[1].set_xlabel('Урожайность (т/га)')
    axes[1].set_ylabel('Плотность')
    axes[1].set_title('Распределение урожайности')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Улучшения
    improvements = [
        results['improvement_vs_baseline'],
        results['improvement_vs_uniform']
    ]
    stds = [
        results['std_improvement_vs_baseline'],
        results['std_improvement_vs_uniform']
    ]
    
    bars = axes[2].bar(['vs Базовый', 'vs Единые дозы'], improvements, 
                       yerr=stds, capsize=5, alpha=0.7)
    axes[2].axhline(y=0, color='red', linestyle='-', linewidth=0.5)
    axes[2].set_ylabel('Улучшение урожайности (%)')
    axes[2].set_title('Относительное улучшение урожайности')
    
    # Добавляем значения на столбцы
    for bar, imp in zip(bars, improvements):
        height = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width()/2., height,
                    f'{imp:.1f}%', ha='center', va='bottom')
    
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{save_dir}/yield_comparison.png')
    plt.show()

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Загружаем данные
    train_dataset, val_dataset, scaler_X, scaler_y, feature_names = prepare_data(
        'data/raw/features.csv',
        'data/raw/targets.csv'
    )
    
    # Загружаем модель
    input_dim = train_dataset.features.shape[1]
    model = MultiOutputMLP(input_dim)
    model.load_state_dict(torch.load('models/best_model.pth', map_location=device))
    model.to(device)
    
    # Оцениваем точность предсказаний
    print("Оценка точности предсказаний доз удобрений...")
    metrics, preds, targets = evaluate_predictions(model, val_dataset, scaler_y, device)
    
    print("\n" + "="*60)
    print("ТОЧНОСТЬ ПРЕДСКАЗАНИЯ ДОЗ УДОБРЕНИЙ")
    print("="*60)
    for nutrient, metric in metrics.items():
        print(f"{nutrient}:")
        print(f"  MAE: {metric['mae']:.2f} кг/га")
        print(f"  RMSE: {metric['rmse']:.2f} кг/га")
        print(f"  MAPE: {metric['mape']:.1f}%")
    
    # Визуализируем качество рекомендаций
    plot_recommendation_analysis(preds, targets, ['N', 'P', 'K'])
    
    # Симулируем урожайность
    print("\n" + "="*60)
    print("СРАВНЕНИЕ УРОЖАЙНОСТИ ПРИ РАЗНЫХ СТРАТЕГИЯХ")
    print("="*60)
    
    # Загружаем исходные признаки для симуляции
    features_df = pd.read_csv('data/raw/features.csv')
    
    results, yields_model, yields_baseline, yields_uniform = simulate_yield_comparison(
        model, val_dataset, scaler_y, features_df, device
    )
    
    print(f"\nСредняя урожайность:")
    print(f"  Рекомендации модели: {results['mean_yield_model']:.2f} т/га")
    print(f"  Базовые дозы: {results['mean_yield_baseline']:.2f} т/га")
    print(f"  Единые дозы: {results['mean_yield_uniform']:.2f} т/га")
    
    print(f"\nУлучшение:")
    print(f"  Относительно базовых доз: +{results['improvement_vs_baseline']:.1f}%")
    print(f"  Относительно единых доз: +{results['improvement_vs_uniform']:.1f}%")
    
    # Визуализируем сравнение
    plot_yield_comparison(results, yields_model, yields_baseline, yields_uniform)
    
    # Сохраняем результаты
    final_results = {
        'prediction_metrics': metrics,
        'yield_comparison': results
    }
    
    with open('reports/final_evaluation.json', 'w') as f:
        # Преобразуем numpy значения в обычные float
        def convert_to_serializable(obj):
            if isinstance(obj, np.floating):
                return float(obj)
            if isinstance(obj, np.integer):
                return int(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        results_serializable = json.loads(
            json.dumps(final_results, default=convert_to_serializable)
        )
        json.dump(results_serializable, f, indent=4)
    
    print("\n✅ Оценка завершена! Результаты сохранены в 'reports/'")

if __name__ == "__main__":
    main()