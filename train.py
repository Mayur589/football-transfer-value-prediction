import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Creating 'models' and 'static' directories if they don't exist...")
os.makedirs('models', exist_ok=True)
os.makedirs('static', exist_ok=True)
print("Loading dataset from final_data.csv...")
try:
    df = pd.read_csv('./data/final_data.csv') 
except FileNotFoundError:
    print("Error: final_data.csv not found. Please make sure it's in the same directory.")
    exit()

columns_to_drop = ['player', 'team', 'name', 'position']
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)
df.fillna(df.median(numeric_only=True), inplace=True)

selected_features = ['highest_value', 'minutes played', 'appearance', 'award', 'assists', 'goals', 'position_encoded',
                     'goals conceded', 'clean sheets', 'yellow cards', 'second yellow cards', 'red cards',
                     'days_injured', 'games_injured', 'height', 'age', 'winger']

target = 'current_value'

available_features = [col for col in selected_features if col in df.columns]
missing_features = [col for col in selected_features if col not in df.columns]

if missing_features:
    print(f"Warning: The following features from your list were NOT found in the CSV: {missing_features}")

if not available_features:
     print("Error: None of the selected features were found in the dataset. Exiting.")
     exit()

X = df[available_features]
y = np.log(df[target] + 1)

print(f"Using {len(available_features)} selected features: {available_features}")


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


y_test_orig = np.exp(y_test) - 1


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

joblib.dump(scaler, 'models/scaler.joblib')
print("Scaler saved to models/scaler.joblib")

models = {
    'LinearRegression': LinearRegression(),
    'Ridge': Ridge(random_state=42),
    'Lasso': Lasso(random_state=42),
    'RandomForest': RandomForestRegressor(random_state=42, n_jobs=-1),
    'GradientBoosting': GradientBoostingRegressor(random_state=42),
    'ExtraTrees': ExtraTreesRegressor(random_state=42, n_jobs=-1),
    'XGBoost': xgb.XGBRegressor(random_state=42, n_jobs=-1),
    'MLP': MLPRegressor(random_state=42, max_iter=1000)
}

params = {
    'LinearRegression': {
       # check if model is fitted with simple linear regression 
    },
    'Ridge': {
        'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    'Lasso': {
        'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]
    },
    'RandomForest': {
        'n_estimators': [400, 600, 800],
        'max_depth': [15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5]
    },
    'GradientBoosting': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'subsample': [0.8, 0.9, 1.0]
    },
    'ExtraTrees': {
        'n_estimators': [400, 600, 800],
        'max_depth': [15, 20, 25, 30, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', 0.5]
    },
    'XGBoost': {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [3, 5, 7],
        'min_child_weight': [1, 3, 5],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0]
    },
    'MLP': { 
        'hidden_layer_sizes': [(64, 32), (100, 50), (128, 64, 32)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.001, 0.01], 
        'learning_rate_init': [0.001, 0.005]
    }
}

results = {}
all_predictions = {}
best_model = None
best_model_name = ""
best_score = float('inf') 

for name, model in models.items():
    print(f"\n--- Training {name} ---")
    
    if name in ['MLP', 'SVR']:
        X_tr, X_te = X_train_scaled, X_test_scaled
    else:
        X_tr, X_te = X_train, X_test
    
    print(f"Hyperparameter tuning with RandomizedSearchCV...")
    n_iter = 100 
    
    if name in ['LinearRegression']:
        model_fitted = model.fit(X_tr, y_train)
        print(f"Using default parameters for {name}")
    else:
        grid = RandomizedSearchCV(model, params[name], n_iter=n_iter, cv=5, 
                                  scoring='neg_mean_squared_error', 
                                  verbose=1, n_jobs=-1, random_state=42)
        grid.fit(X_tr, y_train)
        model_fitted = grid.best_estimator_
        print(f"Best params for {name}: {grid.best_params_}")
    
    y_pred_log = model_fitted.predict(X_te)
    
    rmse_log = np.sqrt(mean_squared_error(y_test, y_pred_log))
    mae_log = mean_absolute_error(y_test, y_pred_log)
    r2 = r2_score(y_test, y_pred_log)
    

    y_pred_orig = np.exp(y_pred_log) - 1
    

    all_predictions[name] = y_pred_orig
    
    rmse_orig = np.sqrt(mean_squared_error(y_test_orig, y_pred_orig))
    mae_orig = mean_absolute_error(y_test_orig, y_pred_orig)
    
    results[name] = {'R2': r2, 'MAE_log': mae_log, 'RMSE_log': rmse_log, 'MAE_orig': mae_orig, 'RMSE_orig': rmse_orig}
    
    print(f"\n--- Results for {name} ---")
    print(f"  R-squared (R2):        {r2:.4f}")
    print(f"  Log-Scale MAE:         {mae_log:.4f} (Error in log(value))")
    print(f"  Original-Scale MAE:    ${mae_orig:,.2f} (Average error in currency)")
    print(f"  Original-Scale RMSE:   ${rmse_orig:,.2f} (Root mean error in currency)")
    
    if mae_orig < best_score:
        best_score = mae_orig
        best_model = model_fitted
        best_model_name = name

print(f"\n========================================================")
print(f"🏆 Best Model: {best_model_name} (MAE: ${best_score:,.2f})")
print(f"========================================================")

print(f"\nSaving best model ({best_model_name}) to models/best_model.joblib...")
joblib.dump(best_model, f'models/best_model.joblib')
print("Model saved.")

if hasattr(best_model, 'feature_importances_'):
    print("Generating feature importance plot...")
    importances = best_model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.title(f'Feature Importances for {best_model_name}')
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [X.columns[i] for i in indices], rotation=60, ha='right')
    plt.tight_layout()
    plt.savefig('static/feature_importance.png')
    plt.close()
    print("Feature importance plot saved to static/feature_importance.png")
else:
    print(f"Cannot create feature importance plot (model type {best_model_name} doesn't support it directly).")

print("\nGenerating model comparison plots...")
sns.set_theme(style="whitegrid")

results_df = pd.DataFrame(results).T.reset_index().rename(columns={'index': 'Model'})

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MAE_orig', data=results_df.sort_values('MAE_orig', ascending=True), palette='Reds_d')
plt.title('Model Comparison: Mean Absolute Error (Original Scale)', fontsize=16, fontweight='bold')
plt.ylabel('Mean Absolute Error (in currency)')
plt.xlabel('Model')
plt.tight_layout()
plt.savefig('static/comparison_mae_original.png')
plt.close()
print("Saved: static/comparison_mae_original.png")

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='RMSE_orig', data=results_df.sort_values('RMSE_orig', ascending=True), palette='Blues_d')
plt.title('Model Comparison: Root Mean Squared Error (Original Scale)', fontsize=16, fontweight='bold')
plt.ylabel('RMSE (in currency)')
plt.xlabel('Model')
plt.tight_layout()
plt.savefig('static/comparison_rmse_original.png')
plt.close()
print("Saved: static/comparison_rmse_original.png")

plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R2', data=results_df.sort_values('R2', ascending=False), palette='Greens_d')
plt.title('Model Comparison: R-squared (R2) Score', fontsize=16, fontweight='bold')
plt.ylabel('R-squared (R2) Score')
plt.xlabel('Model')
plt.tight_layout()
plt.savefig('static/comparison_r2_score.png')
plt.close()
print("Saved: static/comparison_r2_score.png")

def plot_pred_vs_actual(y_true, y_pred, model_name, file_path):
    plt.figure(figsize=(10, 10))
    max_val = max(y_true.max(), y_pred.max()) * 1.05
    min_val = min(y_true.min(), y_pred.min()) * 0.95
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    sns.scatterplot(x=y_true, y=y_pred, alpha=0.6, s=50)
    
    plt.title(f'{model_name}: Predicted vs. Actual Values', fontsize=16, fontweight='bold')
    plt.xlabel('Actual Value (Original Scale)')
    plt.ylabel('Predicted Value (Original Scale)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(file_path)
    plt.close()
    print(f"Saved: {file_path}")

sorted_models = sorted(results.items(), key=lambda x: x[1]['MAE_orig'])
top_3_models = [model_name for model_name, _ in sorted_models[:3]]

for model_name in top_3_models:
    if model_name in all_predictions:
        plot_pred_vs_actual(y_test_orig, all_predictions[model_name], 
                           model_name, f'static/scatter_{model_name.lower().replace(" ", "_")}.png')


print("\nProcess finished.")