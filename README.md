# Football Transfer Value Predictor

A machine learning web application that predicts football player transfer values using XGBoost.

## Features

- Predicts player transfer values based on 17 carefully selected features
- Web interface built with Flask
- Enter per-90-minute stats as non-negative decimals with any precision (e.g., 0.009 is valid)
- Fields auto-fill with realistic defaults or zero (no more blank/black fields)
- Compares multiple ML models: LinearRegression, Ridge, Lasso, RandomForest, GradientBoosting, ExtraTrees, XGBoost, MLP
- Feature importance visualization
- Uses statistics transformed to per-90-minutes basis for better accuracy

## Installation

1. Clone the repository
2. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Train the model (if not already done):
   ```bash
   python train.py
   ```

## Running the Application

### Method 1: Using the shell script
```bash
chmod +x run.sh
./run.sh
```

### Method 2: Manually
```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python app.py
```

The application will be available at `http://localhost:5001`

## Project Structure

```
transfer/
├── app.py              # Flask web application
├── train.py            # Model training script
├── requirements.txt    # Python dependencies
├── data/
│   └── final_data.csv # Training data
├── models/             # Trained models
│   ├── "all the trained models" # auto-generated from the train.py
├── static/             # Static files
│   └── "all the comparisions photos"
└── templates/          # HTML templates
    ├── index.html      # Main form
    └── result.html     # Results page
```

## Model Features

The model uses the following 17 features (ordered by importance):
1. Highest Value (EUR) - Player's peak market value
2. Total Minutes Played - Career minutes
3. Total Appearances - Number of games played
4. Total Awards - Career awards/trophies
5. Assists (per 90 minutes) - Assists per game
6. Goals (per 90 minutes) - Goals per game
7. Position Encoded (1=GK, 2=Defender, 3=Midfielder, 4=Attacker)
8. Goals Conceded (per 90 min)
9. Clean Sheets (per 90 min)
10. Yellow Cards (per 90 min)
11. Second Yellow Cards (per 90 min)
12. Red Cards (per 90 min)
13. Days Injured
14. Games Injured
15. Height (cm)
16. Age
17. Is Winger (0=No, 1=Yes)

**Frontend Note**: All per-90-minute fields now accept numbers like 0.000345 or 0.01 (not just two decimals). You can paste or enter numbers of arbitrary length for detailed stats. Presets and manual entry now prevent blank or black fields; any empty field will be set to 0 automatically for user clarity.

## Model Performance

### Model Comparison (November 2025)

After running `train.py` with all algorithm options, these results were achieved:

- **LinearRegression**: MAE $18,179,005.93, RMSE $337,335,963.82, R² 0.3114
- **Ridge**: MAE $18,112,885.89, RMSE $337,236,581.50, R² 0.3119
- **Lasso**: MAE $18,116,098.43, RMSE $337,054,204.26, R² 0.3118
- **RandomForest**: MAE $1,055,448.05, RMSE $3,315,894.17, R² 0.8476
- **GradientBoosting**: MAE $1,032,581.42, RMSE $3,128,334.91, R² 0.8461 (**Best Model**)
- **ExtraTrees**: MAE $1,192,643.23, RMSE $3,659,113.15, R² 0.7902
- **XGBoost**: MAE $1,051,727.26, RMSE $3,206,929.13, R² 0.8454
- **MLP (Neural Net)**: MAE $1,305,825.23, RMSE $5,144,672.07, R² 0.6253

Hyperparameters for each model were tuned via RandomizedSearchCV. The best model this run was GradientBoosting, achieving:

- **MAE**: $1,032,581.42
- **RMSE**: $3,128,334.91
- **R²**: 0.8461 (84.6% of variance explained)

See `static/comparison_*` and `static/scatter_*` images for plots.

## Technologies Used

- Python 3.13
- Flask - Web framework
- XGBoost - Gradient boosting model
- scikit-learn - Machine learning utilities
- pandas - Data processing
- Bootstrap 5 - Frontend styling

## Dataset Information

This dataset was scraped from Transfermarkt on June 10, 2023. Statistics were transformed to per-90-minutes basis (dividing each stat by minutes played/90) for better player comparison. Features were selected based on correlation analysis with the target variable.

## Notes

- The model uses log-transformed target values for better performance
- Statistics are normalized to per-90-minutes basis
- Predictions have a confidence interval of ±15%
- The application runs on port 5001 by default

# football-transfer-value-prediction
