import json
import pandas as pd
from pathlib import Path

def normalize_weather(raw_json: dict) -> pd.DataFrame:
    """
    Transforms raw weather JSON into a cleaned DataFrame.
    """
    # Extract the daily list from the JSON structure
    df = pd.DataFrame(raw_json.get("daily", []))
    
    if not df.empty:
        # Convert the 'date' column to datetime objects
        df['date'] = pd.to_datetime(df['date'])
        
        # Handle missing values by filling or dropping
        df = df.fillna(0) 
        
    return df

def normalize_source2(raw_json: dict) -> pd.DataFrame:
    """
    Transforms raw Reddit JSON into a cleaned DataFrame aggregated by date.
    """
    # Navigate the Reddit JSON structure: data -> children -> data
    posts = [post['data'] for post in raw_json.get('data', {}).get('children', [])]
    df = pd.DataFrame(posts)
    
    if not df.empty:
        # Convert 'created_utc' timestamp to datetime objects
        df['date'] = pd.to_datetime(df['created_utc'], unit='s').dt.normalize()
        
        # Select relevant columns and handle missing values
        # We aggregate by date to match the weather data's granularity
        df = df[['date', 'ups', 'num_comments']].fillna(0)
        df = df.groupby('date').agg({'ups': 'sum', 'num_comments': 'sum'}).reset_index()
        
    return df

def main():
    # Setup paths
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    weather_path = data_dir / "raw_weather.json"
    reddit_path = data_dir / "raw_source2.json"
    output_path = data_dir / "processed.csv"

    # Load raw data
    with open(weather_path, "r", encoding="utf-8") as f:
        weather_raw = json.load(f)
    with open(reddit_path, "r", encoding="utf-8") as f:
        reddit_raw = json.load(f)

    # Step 1: Normalize both sources
    print("Normalizing weather data...")
    df_weather = normalize_weather(weather_raw)
    
    print("Normalizing Reddit data...")
    df_source2 = normalize_source2(reddit_raw)

    # Step 2: Merge data on the 'date' column using an inner join
    print("Merging datasets...")
    df_merged = df_weather.merge(df_source2, on="date", how="inner")

    # Step 3: Save the final result to CSV
    df_merged.to_csv(output_path, index=False)
    print(f"Success! Processed data saved to {output_path}")
    print(df_merged.head())

if __name__ == "__main__":
    main()