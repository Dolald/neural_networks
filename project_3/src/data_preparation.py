import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

class FertilizerDataGenerator:
    """
    Генератор синтетических данных для рекомендаций удобрений
    """
    def __init__(self, n_samples=10000, random_state=42):
        self.n_samples = n_samples
        np.random.seed(random_state)
        
    def generate_soil_data(self):
        """
        Генерация данных о почве
        """
        # Почвенные характеристики
        soil_types = ['clay', 'loam', 'sandy', 'silty']
        soil_type = np.random.choice(soil_types, self.n_samples)
        
        # Числовые характеристики почвы
        data = {
            'soil_type': soil_type,
            'pH': np.random.normal(6.5, 1.0, self.n_samples).clip(4, 9),
            'N_soil': np.random.normal(50, 20, self.n_samples).clip(0, 200),
            'P_soil': np.random.normal(30, 15, self.n_samples).clip(0, 150),
            'K_soil': np.random.normal(150, 50, self.n_samples).clip(0, 500),
            'organic_matter': np.random.normal(3, 1.5, self.n_samples).clip(0, 10),
            'cec': np.random.normal(20, 10, self.n_samples).clip(0, 50)
        }
        return pd.DataFrame(data)
    
    def generate_weather_data(self):
        """
        Генерация погодных данных
        """
        data = {
            'temperature': np.random.normal(20, 10, self.n_samples).clip(-5, 40),
            'precipitation': np.random.exponential(50, self.n_samples).clip(0, 300),
            'humidity': np.random.normal(60, 20, self.n_samples).clip(0, 100),
            'sunshine_hours': np.random.normal(8, 3, self.n_samples).clip(0, 16)
        }
        return pd.DataFrame(data)
    
    def generate_crop_data(self):
        """
        Генерация данных о культуре
        """
        crop_types = ['wheat', 'corn', 'rice', 'soybean', 'potato']
        crop_type = np.random.choice(crop_types, self.n_samples)
        
        data = {
            'crop_type': crop_type,
            'previous_yield': np.random.normal(5, 2, self.n_samples).clip(0, 15),
            'planting_density': np.random.normal(300, 100, self.n_samples).clip(50, 600)
        }
        return pd.DataFrame(data)
    
    def generate_fertilizer_recommendations(self, soil_df, weather_df, crop_df):
        """
        Генерация целевых переменных (рекомендаций по удобрениям)
        на основе входных данных
        """
        # Извлекаем числовые признаки
        pH = soil_df['pH'].values
        N_soil = soil_df['N_soil'].values
        P_soil = soil_df['P_soil'].values
        K_soil = soil_df['K_soil'].values
        om = soil_df['organic_matter'].values
        temp = weather_df['temperature'].values
        precip = weather_df['precipitation'].values
        
        # Создаем зависимость для рекомендаций
        # N рекомендация зависит от N в почве, органики и прошлого урожая
        N_rec = np.maximum(0, 
            150 - 0.5 * N_soil + 
            5 * om + 
            0.1 * precip +
            np.random.normal(0, 10, self.n_samples)
        )
        
        # P рекомендация зависит от P в почве и pH
        P_rec = np.maximum(0,
            80 - 0.3 * P_soil +
            5 * (7 - np.abs(pH - 7)) +
            np.random.normal(0, 5, self.n_samples)
        )
        
        # K рекомендация зависит от K в почве и температуры
        K_rec = np.maximum(0,
            200 - 0.2 * K_soil +
            2 * (temp - 20) +
            np.random.normal(0, 8, self.n_samples)
        )
        
        return pd.DataFrame({
            'N_recommendation': N_rec,
            'P_recommendation': P_rec,
            'K_recommendation': K_rec
        })
    
    def generate_yield_data(self, soil_df, weather_df, crop_df, fert_df):
        """
        Генерация данных об урожайности при разных дозах удобрений
        (симулятор для оценки рекомендаций)
        """
        # Базовая урожайность зависит от почвы и погоды
        base_yield = (
            3 +  # базовое значение
            0.5 * (soil_df['pH'] - 6.5) +
            0.02 * soil_df['organic_matter'] * 10 +
            0.1 * (weather_df['temperature'] - 20) +
            0.005 * weather_df['precipitation']
        )
        
        # Эффект от удобрений (закон убывающей отдачи)
        N_effect = 0.03 * fert_df['N_recommendation'] - 0.0001 * fert_df['N_recommendation']**2
        P_effect = 0.02 * fert_df['P_recommendation'] - 0.0001 * fert_df['P_recommendation']**2
        K_effect = 0.015 * fert_df['K_recommendation'] - 0.00005 * fert_df['K_recommendation']**2
        
        # Общая урожайность
        total_yield = base_yield + N_effect + P_effect + K_effect + np.random.normal(0, 0.5, self.n_samples)
        
        return np.maximum(0, total_yield)

def create_dataset(n_samples=10000):
    """
    Создание полного датасета
    """
    generator = FertilizerDataGenerator(n_samples)
    
    # Генерируем данные
    soil_df = generator.generate_soil_data()
    weather_df = generator.generate_weather_data()
    crop_df = generator.generate_crop_data()
    fert_df = generator.generate_fertilizer_recommendations(soil_df, weather_df, crop_df)
    yield_data = generator.generate_yield_data(soil_df, weather_df, crop_df, fert_df)
    
    # Объединяем все данные
    # Кодируем категориальные переменные
    soil_encoded = pd.get_dummies(soil_df, columns=['soil_type'], prefix='soil')
    crop_encoded = pd.get_dummies(crop_df, columns=['crop_type'], prefix='crop')
    
    # Убираем оригинальные категориальные колонки
    soil_numeric = soil_encoded.drop(columns=[col for col in soil_encoded.columns if 'soil_type' in col and len(col.split('_')) == 2])
    crop_numeric = crop_encoded.drop(columns=[col for col in crop_encoded.columns if 'crop_type' in col and len(col.split('_')) == 2])
    
    # Объединяем все признаки
    features = pd.concat([
        soil_numeric,
        soil_encoded[[col for col in soil_encoded.columns if 'soil_' in col]],
        weather_df,
        crop_numeric,
        crop_encoded[[col for col in crop_encoded.columns if 'crop_' in col]],
    ], axis=1)
    
    targets = fert_df
    targets['yield'] = yield_data
    
    return features, targets

def analyze_data(features, targets):
    """
    Анализ созданных данных
    """
    print("=" * 60)
    print("АНАЛИЗ ДАННЫХ")
    print("=" * 60)
    
    print(f"\nКоличество образцов: {len(features)}")
    print(f"Количество признаков: {features.shape[1]}")
    
    print("\nСтатистика целевых переменных:")
    print(targets.describe())
    
    # Визуализация распределений
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.ravel()
    
    for i, col in enumerate(['N_recommendation', 'P_recommendation', 'K_recommendation', 'yield']):
        axes[i].hist(targets[col], bins=50, edgecolor='black', alpha=0.7)
        axes[i].set_title(f'Distribution of {col}')
        axes[i].set_xlabel(col)
        axes[i].set_ylabel('Frequency')
    
    # Корреляции признаков с целевыми
    corr_with_N = features.corrwith(targets['N_recommendation']).sort_values(ascending=False)
    corr_with_P = features.corrwith(targets['P_recommendation']).sort_values(ascending=False)
    corr_with_K = features.corrwith(targets['K_recommendation']).sort_values(ascending=False)
    
    axes[4].barh(range(5), corr_with_N.head(5).values)
    axes[4].set_yticks(range(5))
    axes[4].set_yticklabels(corr_with_N.head(5).index)
    axes[4].set_title('Top correlations with N recommendation')
    
    axes[5].barh(range(5), corr_with_K.head(5).values)
    axes[5].set_yticks(range(5))
    axes[5].set_yticklabels(corr_with_K.head(5).index)
    axes[5].set_title('Top correlations with K recommendation')
    
    plt.tight_layout()
    plt.savefig('reports/data_analysis.png')
    plt.show()

if __name__ == "__main__":
    # Создаем папки
    os.makedirs('data/raw', exist_ok=True)
    os.makedirs('reports', exist_ok=True)
    
    # Генерируем данные
    print("Генерация данных...")
    features, targets = create_dataset(10000)
    
    # Сохраняем
    features.to_csv('data/raw/features.csv', index=False)
    targets.to_csv('data/raw/targets.csv', index=False)
    
    print(f"Данные сохранены:")
    print(f"  - features.csv: {features.shape}")
    print(f"  - targets.csv: {targets.shape}")
    
    # Анализируем
    analyze_data(features, targets)