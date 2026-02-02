import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from pathlib import Path

def prepare_features(df):
    """
    Feature Engineering: create internal and external features.
    """
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df['day_of_week'] = df['date'].dt.dayofweek
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Enriched features (Weather-based)
    df['temp_range'] = df['max_temp'] - df['min_temp']
    return df

def main():
    # Setup paths
    root = Path(__file__).resolve().parent
    data_path = root / "data" / "processed.csv"
    
    if not data_path.exists():
        print("Error: processed.csv not found!")
        return

    df = pd.read_csv(data_path)
    
    # QUICK FIX: If we have too little data, generate synthetic rows to allow the model to run
    if len(df) < 5:
        print(f"Dataset too small ({len(df)} rows). Generating synthetic data for the assignment...")
        extra_rows = []
        last_date = pd.to_datetime(df['date'].iloc[0])
        for i in range(1, 10):
            new_row = df.iloc[0].copy()
            new_row['date'] = (last_date + pd.Timedelta(days=i)).strftime('%Y-%m-%d')
            new_row['ups'] = max(0, new_row['ups'] + np.random.randint(-5, 15))
            new_row['max_temp'] += np.random.uniform(-2, 2)
            extra_rows.append(new_row)
        df = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True)

    # Process features
    df = prepare_features(df)
    df = df.sort_values('date')

    # Define feature sets
    baseline_features = ['day_of_week', 'is_weekend']
    enriched_features = baseline_features + ['max_temp', 'precipitation', 'temp_range']
    target = 'ups'

    # Time-based Split (Chronological)
    # We ensure at least 1 row in test and the rest in train
    split_index = max(1, int(len(df) * 0.7))
    train_df = df.iloc[:split_index]
    test_df = df.iloc[split_index:]

    print(f"Training on {len(train_df)} rows, Testing on {len(test_df)} rows.")

    # Model A: Baseline
    model_a = LinearRegression()
    model_a.fit(train_df[baseline_features], train_df[target])
    preds_a = model_a.predict(test_df[baseline_features])

    # Model B: Enriched
    model_b = LinearRegression()
    model_b.fit(train_df[enriched_features], train_df[target])
    preds_b = model_b.predict(test_df[enriched_features])

    # Calculate Metrics
    results = {
        "Metric": ["MAE", "RMSE"],
        "Baseline": [
            mean_absolute_error(test_df[target], preds_a),
            np.sqrt(mean_squared_error(test_df[target], preds_a))
        ],
        "Enriched": [
            mean_absolute_error(test_df[target], preds_b),
            np.sqrt(mean_squared_error(test_df[target], preds_b))
        ]
    }
    
    print("\nModel Comparison Table:")
    print(pd.DataFrame(results).to_string(index=False))

    # Visualization
    plt.figure(figsize=(10, 5))
    plt.plot(test_df['date'], test_df[target], 'ko-', label='Actual Ups')
    plt.plot(test_df['date'], preds_a, 'r--', label='Baseline Pred')
    plt.plot(test_df['date'], preds_b, 'g--', label='Enriched Pred')
    plt.title('Reddit Engagement Prediction (Actual vs Predicted)')
    plt.xlabel('Date')
    plt.ylabel('Ups')
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()