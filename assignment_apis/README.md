# 📊 Meteorological Impact on Data Science Community Engagement

## 📝 Executive Summary

This project investigates whether environmental factors (weather) have a measurable correlation with social media activity within the professional data science community. By integrating historical weather data with Reddit engagement metrics, we built a predictive model to determine if "rainy day" behavior translates to higher digital interaction.

---

## 🛠 Tech Stack

- **Data Collection**: `Requests` (REST APIs)
- **Data Processing**: `Pandas`, `NumPy`
- **Machine Learning**: `Scikit-Learn` (Linear Regression)
- **Visualization**: `Matplotlib`

---

## 🛰 Data Pipeline & Architecture

### 1. Data Acquisition

We utilized two primary public data sources:

- **Weather Data (Open-Meteo)**: We transitioned from OpenWeatherMap to Open-Meteo to provide a robust, keyless, and credit-card-free solution for historical weather metrics.
- **Social Media (Reddit)**: We accessed the `r/datascience` subreddit's top posts via its public JSON endpoint. As required by Reddit's API policy, a custom `User-Agent` (`DataScienceCourse/1.0`) was implemented to ensure stable connectivity.

### 2. Normalization & Integration (`normalize.py`)

To prepare the data for analysis, we performed the following:

- **Temporal Alignment**: All timestamps were normalized into a standard `datetime` format.
- **Aggregation**: Reddit data was grouped by date, summing "Ups" and "Comments" to match the daily granularity of the weather data.
- **The Join**: We executed an **Inner Merge** on the `date` column, ensuring the resulting `processed.csv` only contains high-fidelity, overlapping data points.

---

## 🤖 Machine Learning Analysis

### Model Definition

We opted for a **Regression Task** to predict the continuous variable of `ups` (engagement).

### Experimental Setup

We compared two distinct models to evaluate the added value of external data:

1. **Baseline Model**: Uses only "Internal" features—Day of the week and Weekend/Weekday status.
2. **Enriched Model**: Incorporates "External" meteorological features—Maximum temperature and precipitation levels.

### Evaluation Strategy

- **Validation**: A chronological **Time Split** was used (70% Train / 30% Test) instead of a random split to maintain the integrity of time-series data.
- **Metrics**: Accuracy was measured using **Mean Absolute Error (MAE)** and **Root Mean Squared Error (RMSE)**.

---

## 📈 Results & Conclusions

The visualization in `Figure_1.png` demonstrates the model's performance.

- **Observation**: The Enriched model typically tracks closer to the Actual Ups than the Baseline, suggesting that weather patterns do provide subtle predictive signals for community engagement.
- **Outcome**: Lower error metrics in the Enriched model justify the integration of heterogeneous data sources.

---

## 🤖 AI & Copilot Interaction Log

As per assignment requirements, the following interactions assisted in development:

1. **Debugging Authentication**:
   - _Issue_: 401 error with API key.
   - _Resolution_: Switched to Open-Meteo as a free alternative.
2. **Code Generation**:
   - _Prompt_: "Generate normalization code for two JSON files based on a date column."
   - _Action_: Implemented `pd.merge(how='inner')` structure.
3. **Refining Logic**:
   - _Action_: Added manual date parsing to handle Reddit's UTC timestamps correctly.

---

## 📂 Project Structure

```text
├── data/
│   ├── raw_weather.json    # Original weather data
│   ├── raw_source2.json    # Original Reddit data
│   └── processed.csv       # Merged dataset
├── weather_fetcher.py      # Script 1: Weather API logic
├── reddit_fetcher.py       # Script 2: Reddit API logic
├── normalize.py            # Script 3: Cleaning & Merging
├── predict_engagement.py   # Script 4: ML & Visualization
└── README.md               # This documentation
```
