#!/usr/bin/env python3
"""
House Price Prediction Project
===============================

This script demonstrates a complete regression pipeline for 
predicting house prices using various machine learning algorithms.

Author: ML Portfolio
Date: 2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

class HousePricePredictor:
    """
    A complete house price prediction system.
    """
    
    def __init__(self):
        self.models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
        }
        self.preprocessor = None
        self.results = {}
        
    def load_data(self, file_path):
        """Load and return the dataset."""
        try:
            self.data = pd.read_csv(file_path)
            print(f"Data loaded successfully: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print("Creating sample house price data for demonstration...")
            return self.create_sample_data()
    
    def create_sample_data(self):
        """Create sample house price data for demonstration."""
        np.random.seed(42)
        n_samples = 1460
        
        # Generate synthetic features
        data = {
            'OverallQual': np.random.randint(1, 11, n_samples),
            'GrLivArea': np.random.randint(600, 5000, n_samples),
            'GarageArea': np.random.randint(0, 1500, n_samples),
            'TotalBsmtSF': np.random.randint(0, 3000, n_samples),
            'YearBuilt': np.random.randint(1872, 2010, n_samples),
            'YearRemodAdd': np.random.randint(1950, 2010, n_samples),
            'MasVnrArea': np.random.randint(0, 1600, n_samples),
            'BsmtFinSF1': np.random.randint(0, 2000, n_samples),
            'LotArea': np.random.randint(1300, 215000, n_samples),
            'WoodDeckSF': np.random.randint(0, 800, n_samples),
            'OpenPorchSF': np.random.randint(0, 500, n_samples),
            'EnclosedPorch': np.random.randint(0, 400, n_samples),
            'Neighborhood': np.random.choice(['CollgCr', 'Veenker', 'Crawfor', 'NoRidge', 'Mitchel', 
                                            'Somerst', 'NWAmes', 'OldTown', 'BrkSide', 'Sawyer',
                                            'NridgHt', 'NAmes', 'SawyerW', 'IDOTRR', 'MeadowV'], n_samples),
            'MSSubClass': np.random.choice([20, 30, 40, 45, 50, 60, 70, 75, 80, 85, 90, 120, 160, 180, 190], n_samples),
            'MSZoning': np.random.choice(['RL', 'RM', 'C (all)', 'FV', 'RH'], n_samples),
            'BldgType': np.random.choice(['1Fam', '2fmCon', 'Duplex', 'TwnhsE', 'Twnhs'], n_samples),
            'HouseStyle': np.random.choice(['2Story', '1Story', '1.5Fin', '1.5Unf', 'SFoyer', 'SLvl', '2.5Unf', '2.5Fin'], n_samples),
            'RoofStyle': np.random.choice(['Gable', 'Hip', 'Gambrel', 'Mansard', 'Flat', 'Shed'], n_samples),
            'ExterQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa'], n_samples),
            'KitchenQual': np.random.choice(['Ex', 'Gd', 'TA', 'Fa'], n_samples),
            'HeatingQC': np.random.choice(['Ex', 'Gd', 'TA', 'Fa', 'Po'], n_samples),
            'CentralAir': np.random.choice(['Y', 'N'], n_samples),
            'FullBath': np.random.randint(0, 4, n_samples),
            'HalfBath': np.random.randint(0, 3, n_samples),
            'BedroomAbvGr': np.random.randint(0, 8, n_samples),
            'KitchenAbvGr': np.random.randint(0, 4, n_samples),
            'TotRmsAbvGrd': np.random.randint(2, 15, n_samples),
            'Fireplaces': np.random.randint(0, 4, n_samples),
            'GarageCars': np.random.randint(0, 5, n_samples),
            'PoolArea': np.random.randint(0, 800, n_samples),
            'MoSold': np.random.randint(1, 13, n_samples),
            'YrSold': np.random.randint(2006, 2011, n_samples)
        }
        
        # Create target variable with realistic relationships
        df = pd.DataFrame(data)
        
        # Calculate price based on features (realistic relationships)
        base_price = (
            df['OverallQual'] * 15000 +
            df['GrLivArea'] * 50 +
            df['GarageArea'] * 30 +
            df['TotalBsmtSF'] * 25 +
            (2010 - df['YearBuilt']) * -500 +  # Newer houses cost more
            df['LotArea'] * 2 +
            df['FullBath'] * 8000 +
            df['BedroomAbvGr'] * 5000 +
            df['Fireplaces'] * 7000 +
            np.random.normal(0, 15000, n_samples)  # Add some noise
        )
        
        # Add neighborhood premium/discount
        neighborhood_multiplier = {
            'NoRidge': 1.4, 'NridgHt': 1.3, 'Somerst': 1.2, 'Veenker': 1.15,
            'Crawfor': 1.1, 'CollgCr': 1.05, 'NWAmes': 1.0, 'Mitchel': 0.95,
            'NAmes': 0.9, 'Sawyer': 0.85, 'SawyerW': 0.85, 'OldTown': 0.8,
            'BrkSide': 0.75, 'IDOTRR': 0.7, 'MeadowV': 0.65
        }
        
        df['SalePrice'] = base_price * df['Neighborhood'].map(neighborhood_multiplier)
        df['SalePrice'] = np.maximum(df['SalePrice'], 50000)  # Minimum price
        df['SalePrice'] = df['SalePrice'].astype(int)
        
        self.data = df
        print(f"Sample data created: {self.data.shape}")
        return self.data
    
    def explore_data(self):
        """Perform exploratory data analysis."""
        print("\n=== DATA EXPLORATION ===")
        print("\nDataset Info:")
        print(self.data.info())
        
        print("\nDataset Description:")
        print(self.data.describe())
        
        print(f"\nTarget Variable (SalePrice) Statistics:")
        print(f"Mean: ${self.data['SalePrice'].mean():,.0f}")
        print(f"Median: ${self.data['SalePrice'].median():,.0f}")
        print(f"Min: ${self.data['SalePrice'].min():,.0f}")
        print(f"Max: ${self.data['SalePrice'].max():,.0f}")
        
        # Visualizations
        plt.figure(figsize=(20, 15))
        
        # Price distribution
        plt.subplot(3, 4, 1)
        plt.hist(self.data['SalePrice'], bins=50, alpha=0.7, color='skyblue')
        plt.title('Sale Price Distribution')
        plt.xlabel('Sale Price ($)')
        plt.ylabel('Frequency')
        
        # Price vs key features
        numeric_features = ['GrLivArea', 'TotalBsmtSF', 'GarageArea', 'LotArea']
        for i, feature in enumerate(numeric_features, 2):
            plt.subplot(3, 4, i)
            plt.scatter(self.data[feature], self.data['SalePrice'], alpha=0.5)
            plt.title(f'Sale Price vs {feature}')
            plt.xlabel(feature)
            plt.ylabel('Sale Price ($)')
        
        # Categorical features
        categorical_features = ['OverallQual', 'Neighborhood', 'ExterQual', 'KitchenQual']
        for i, feature in enumerate(categorical_features, 6):
            plt.subplot(3, 4, i)
            if feature == 'OverallQual':
                avg_prices = self.data.groupby(feature)['SalePrice'].mean()
                avg_prices.plot(kind='bar')
                plt.title(f'Average Price by {feature}')
                plt.ylabel('Average Sale Price ($)')
            elif feature == 'Neighborhood':
                avg_prices = self.data.groupby(feature)['SalePrice'].mean().sort_values(ascending=False)
                avg_prices.head(10).plot(kind='bar')
                plt.title(f'Top 10 Neighborhoods by Avg Price')
                plt.ylabel('Average Sale Price ($)')
                plt.xticks(rotation=45)
            else:
                avg_prices = self.data.groupby(feature)['SalePrice'].mean()
                avg_prices.plot(kind='bar')
                plt.title(f'Average Price by {feature}')
                plt.ylabel('Average Sale Price ($)')
        
        # Correlation heatmap
        plt.subplot(3, 4, 10)
        numeric_cols = self.data.select_dtypes(include=[np.number]).columns
        correlation_matrix = self.data[numeric_cols].corr()
        
        # Get top correlations with SalePrice
        price_corr = correlation_matrix['SalePrice'].sort_values(ascending=False)
        top_features = price_corr.head(10).index
        
        sns.heatmap(correlation_matrix.loc[top_features, top_features], 
                   annot=True, cmap='coolwarm', center=0, fmt='.2f')
        plt.title('Top Features Correlation with Sale Price')
        
        # Feature importance preview
        plt.subplot(3, 4, 11)
        price_corr_abs = price_corr.abs().sort_values(ascending=False)
        price_corr_abs.head(10).plot(kind='bar')
        plt.title('Top 10 Features by Correlation with Price')
        plt.ylabel('Absolute Correlation')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig('house_price_exploration.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def feature_engineering(self):
        """Create new features and prepare data."""
        print("\n=== FEATURE ENGINEERING ===")
        
        df = self.data.copy()
        
        # Create new features
        df['TotalSF'] = df['TotalBsmtSF'] + df['GrLivArea']
        df['TotalBathrooms'] = df['FullBath'] + 0.5 * df['HalfBath']
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['YearsSinceRemod'] = df['YrSold'] - df['YearRemodAdd']
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
        df['HasBasement'] = (df['TotalBsmtSF'] > 0).astype(int)
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
        
        # Quality-related features
        quality_map = {'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}
        df['ExterQual_num'] = df['ExterQual'].map(quality_map)
        df['KitchenQual_num'] = df['KitchenQual'].map(quality_map)
        df['HeatingQC_num'] = df['HeatingQC'].map(quality_map)
        
        # Interaction features
        df['QualitySize'] = df['OverallQual'] * df['TotalSF']
        df['QualityAge'] = df['OverallQual'] * (2010 - df['YearBuilt'])
        
        # Log transform skewed features
        skewed_features = ['LotArea', 'MasVnrArea', 'BsmtFinSF1', 'WoodDeckSF']
        for feature in skewed_features:
            if feature in df.columns:
                df[f'{feature}_log'] = np.log1p(df[feature])
        
        self.engineered_data = df
        
        # Identify categorical and numerical columns
        categorical_features = df.select_dtypes(include=['object']).columns.tolist()
        numerical_features = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target from numerical features
        if 'SalePrice' in numerical_features:
            numerical_features.remove('SalePrice')
        
        print(f"Created {len(df.columns) - len(self.data.columns)} new features")
        print(f"Categorical features: {len(categorical_features)}")
        print(f"Numerical features: {len(numerical_features)}")
        
        return categorical_features, numerical_features
    
    def preprocess_data(self):
        """Preprocess the data for machine learning."""
        print("\n=== DATA PREPROCESSING ===")
        
        # Feature engineering
        categorical_features, numerical_features = self.feature_engineering()
        
        # Prepare features and target
        X = self.engineered_data.drop(['SalePrice'], axis=1)
        y = self.engineered_data['SalePrice']
        
        # Create preprocessing pipeline
        numerical_transformer = StandardScaler()
        categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse=False)
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        print(f"Training set size: {self.X_train.shape}")
        print(f"Test set size: {self.X_test.shape}")
        print(f"Features used: {len(numerical_features)} numerical + {len(categorical_features)} categorical")
    
    def train_models(self):
        """Train multiple regression models."""
        print("\n=== MODEL TRAINING ===")
        
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline with preprocessing
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('regressor', model)
            ])
            
            # Train the model
            pipeline.fit(self.X_train, self.y_train)
            
            # Make predictions
            y_pred_train = pipeline.predict(self.X_train)
            y_pred_test = pipeline.predict(self.X_test)
            
            # Calculate metrics
            train_rmse = np.sqrt(mean_squared_error(self.y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(self.y_test, y_pred_test))
            train_mae = mean_absolute_error(self.y_train, y_pred_train)
            test_mae = mean_absolute_error(self.y_test, y_pred_test)
            train_r2 = r2_score(self.y_train, y_pred_train)
            test_r2 = r2_score(self.y_test, y_pred_test)
            
            # Cross-validation
            cv_scores = cross_val_score(pipeline, self.X_train, self.y_train, 
                                      cv=5, scoring='neg_root_mean_squared_error')
            cv_rmse = -cv_scores.mean()
            
            # Store results
            self.results[name] = {
                'pipeline': pipeline,
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2,
                'cv_rmse': cv_rmse,
                'predictions': y_pred_test
            }
            
            print(f"Test RMSE: ${test_rmse:,.0f}")
            print(f"Test MAE: ${test_mae:,.0f}")
            print(f"Test R²: {test_r2:.3f}")
            print(f"CV RMSE: ${cv_rmse:,.0f}")
    
    def evaluate_models(self):
        """Evaluate and compare model performance."""
        print("\n=== MODEL EVALUATION ===")
        
        # Create comparison dataframe
        comparison_data = []
        for name, result in self.results.items():
            comparison_data.append({
                'Model': name,
                'Test RMSE': result['test_rmse'],
                'Test MAE': result['test_mae'],
                'Test R²': result['test_r2'],
                'CV RMSE': result['cv_rmse']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_df = comparison_df.sort_values('Test RMSE')
        print("\nModel Comparison (sorted by Test RMSE):")
        print(comparison_df.round(3))
        
        # Plot model comparison
        plt.figure(figsize=(15, 10))
        
        # RMSE comparison
        plt.subplot(2, 3, 1)
        plt.bar(comparison_df['Model'], comparison_df['Test RMSE'])
        plt.title('Model RMSE Comparison')
        plt.ylabel('RMSE ($)')
        plt.xticks(rotation=45)
        
        # R² comparison
        plt.subplot(2, 3, 2)
        plt.bar(comparison_df['Model'], comparison_df['Test R²'])
        plt.title('Model R² Score Comparison')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        
        # Actual vs Predicted for best model
        best_model_name = comparison_df.iloc[0]['Model']
        best_predictions = self.results[best_model_name]['predictions']
        
        plt.subplot(2, 3, 3)
        plt.scatter(self.y_test, best_predictions, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Price ($)')
        plt.ylabel('Predicted Price ($)')
        plt.title(f'Actual vs Predicted - {best_model_name}')
        
        # Residuals plot
        plt.subplot(2, 3, 4)
        residuals = self.y_test - best_predictions
        plt.scatter(best_predictions, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Price ($)')
        plt.ylabel('Residuals ($)')
        plt.title(f'Residuals Plot - {best_model_name}')
        
        # Feature importance (for tree-based models)
        plt.subplot(2, 3, 5)
        if 'Random Forest' in self.results:
            rf_pipeline = self.results['Random Forest']['pipeline']
            rf_model = rf_pipeline.named_steps['regressor']
            
            # Get feature names after preprocessing
            cat_features = rf_pipeline.named_steps['preprocessor'].named_transformers_['cat'].get_feature_names_out()
            num_features = rf_pipeline.named_steps['preprocessor'].named_transformers_['num'].get_feature_names_out()
            all_features = list(num_features) + list(cat_features)
            
            feature_importance = rf_model.feature_importances_
            
            # Get top 10 features
            importance_df = pd.DataFrame({
                'feature': all_features,
                'importance': feature_importance
            }).sort_values('importance', ascending=False).head(10)
            
            plt.barh(range(len(importance_df)), importance_df['importance'])
            plt.yticks(range(len(importance_df)), importance_df['feature'])
            plt.xlabel('Importance')
            plt.title('Top 10 Feature Importance (Random Forest)')
        
        # Model comparison radar chart
        plt.subplot(2, 3, 6)
        models = comparison_df['Model'].tolist()
        r2_scores = comparison_df['Test R²'].tolist()
        plt.plot(models, r2_scores, 'o-', linewidth=2, markersize=8)
        plt.title('R² Score Comparison')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('regression_model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return best_model_name
    
    def analyze_predictions(self, model_name=None):
        """Analyze prediction accuracy in different price ranges."""
        if model_name is None:
            # Use best model
            best_idx = min(self.results.keys(), key=lambda k: self.results[k]['test_rmse'])
            model_name = best_idx
        
        print(f"\n=== PREDICTION ANALYSIS FOR {model_name.upper()} ===")
        
        predictions = self.results[model_name]['predictions']
        
        # Create price range analysis
        price_ranges = [
            (0, 150000, 'Low'),
            (150000, 300000, 'Medium'),
            (300000, 500000, 'High'),
            (500000, float('inf'), 'Luxury')
        ]
        
        for min_price, max_price, category in price_ranges:
            mask = (self.y_test >= min_price) & (self.y_test < max_price)
            if mask.sum() > 0:
                range_actual = self.y_test[mask]
                range_pred = predictions[mask]
                
                rmse = np.sqrt(mean_squared_error(range_actual, range_pred))
                mae = mean_absolute_error(range_actual, range_pred)
                r2 = r2_score(range_actual, range_pred)
                
                print(f"{category} Price Range (${min_price:,} - ${max_price:,}):")
                print(f"  Count: {mask.sum()}")
                print(f"  RMSE: ${rmse:,.0f}")
                print(f"  MAE: ${mae:,.0f}")
                print(f"  R²: {r2:.3f}")
    
    def generate_insights(self):
        """Generate insights from the regression analysis."""
        print("\n=== BUSINESS INSIGHTS ===")
        
        insights = [
            "1. Overall Quality is the most important factor in determining house price",
            "2. Total square footage (living area + basement) strongly correlates with price",
            "3. Neighborhood location creates significant price premiums/discounts",
            "4. Newer houses and recent renovations add substantial value",
            "5. Kitchen and exterior quality are key selling points",
            "6. Garage area and basement space contribute meaningfully to price"
        ]
        
        print("\nKey Insights:")
        for insight in insights:
            print(insight)
        
        print("\nRecommendations for Real Estate:")
        recommendations = [
            "• Focus on overall quality improvements for maximum ROI",
            "• Emphasize total living space in property descriptions",
            "• Consider neighborhood comparables when pricing",
            "• Highlight recent renovations and modern features",
            "• Invest in kitchen and exterior improvements",
            "• Utilize basement and garage space effectively"
        ]
        
        for rec in recommendations:
            print(rec)

def main():
    """Main execution function."""
    print("=== HOUSE PRICE PREDICTION PROJECT ===")
    
    # Initialize predictor
    predictor = HousePricePredictor()
    
    # Load data (will create sample data if file not found)
    predictor.load_data('house_prices.csv')
    
    # Perform analysis
    predictor.explore_data()
    predictor.preprocess_data()
    predictor.train_models()
    best_model = predictor.evaluate_models()
    predictor.analyze_predictions(best_model)
    predictor.generate_insights()
    
    print(f"\n=== ANALYSIS COMPLETE ===")
    print(f"Best performing model: {best_model}")
    print("Check generated plots: house_price_exploration.png, regression_model_evaluation.png")

if __name__ == "__main__":
    main()