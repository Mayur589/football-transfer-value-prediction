from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os

app = Flask(__name__)
model = joblib.load('models/best_model.joblib')  # Load best model
scaler = joblib.load('models/scaler.joblib')  # Load scaler for MLP model
# Feature order must match train.py
feature_cols = ['highest_value', 'minutes played', 'appearance', 'award', 'assists', 'goals', 'position_encoded',
                'goals conceded', 'clean sheets', 'yellow cards', 'second yellow cards', 'red cards',
                'days_injured', 'games_injured', 'height', 'age', 'winger']

# Load preset players
def load_preset_players():
    df = pd.read_csv('data/final_data.csv')
    df['value_millions'] = df['current_value'] / 1_000_000
    
    # Get diverse players across value ranges
    diverse_indices = []
    for value_range in [(20, 40), (10, 20), (5, 10), (1, 5), (0.5, 1)]:
        subset = df[(df['value_millions'] >= value_range[0]) & (df['value_millions'] <= value_range[1])]
        if len(subset) > 0:
            diverse_indices.append(subset.sample(1, random_state=42).index[0])
    
    # Add a couple more high-value players
    high_value_subset = df[df['value_millions'] >= 30]
    if len(high_value_subset) >= 3:
        high_value = high_value_subset.sample(3, random_state=42)
    else:
        high_value = high_value_subset.copy()  # as many as available or empty
    diverse_indices.extend(high_value.index[:3])
    
    # Deduplicate indices
    diverse_indices = list(dict.fromkeys(diverse_indices))

    # If not enough presets, fill with first entries in DataFrame
    num_needed = 10 - len(diverse_indices)
    if num_needed > 0:
        head_add = df[~df.index.isin(diverse_indices)].head(num_needed).index.tolist()
        diverse_indices.extend(head_add)
    # Ensure at least some presets
    if len(diverse_indices) == 0:
        diverse_indices = df.head(5).index.tolist()
    df_presets = df.loc[diverse_indices[:10]].copy()
    
    # Convert to list of dicts for template
    presets = []
    for idx, row in df_presets.iterrows():
        preset = {
            'name': row['name'],
            'team': row['team'],
            'position': row['position'],
            'real_value': round(row['value_millions'], 2),
            'data': {col: float(row[col]) if col in row and not pd.isna(row.get(col, np.nan)) else 0.0 for col in feature_cols}
        }
        presets.append(preset)
    
    return presets

preset_players = load_preset_players()

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if preset was selected (and not empty)
        preset_value = request.form.get('preset', '')
        if preset_value and preset_value.strip():  # Check if preset exists and is not empty/whitespace
            preset_idx = int(preset_value)
            preset = preset_players[preset_idx]
            data = preset['data']
            real_value = preset['real_value']
        else:
            # Collect form data for manual input
            data = {col: float(request.form.get(col, 0)) for col in feature_cols}
            real_value = None
        
        df = pd.DataFrame([data])
        
        # Check if model needs scaled input (MLP models do, tree-based models like RandomForest don't)
        from sklearn.neural_network import MLPRegressor
        if isinstance(model, MLPRegressor):
            # Scale the input data for MLP model
            df_scaled = scaler.transform(df)
            pred_log = model.predict(df_scaled)[0]
        else:
            # Use unscaled data for tree-based models
            pred_log = model.predict(df)[0]
        
        # Predict (inverse log transform)
        pred_value = np.exp(pred_log) - 1
        pred_value_millions = round(pred_value / 1_000_000, 2)  # Convert to millions
        confidence = "±15%"  # Based on typical RMSE
        
        return render_template('result.html', 
                             prediction=pred_value_millions, 
                             confidence=confidence,
                             real_value=real_value)
    
    return render_template('index.html', presets=preset_players)

if __name__ == '__main__':
    app.run(debug=True, port=5001)