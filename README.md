# ✈️ Flight Arrival Delay Predictor

**BSAN 6070 - Introduction to Machine Learning | Spring 2026**
**Loyola Marymount University | College of Business Administration**

> **Team:** Prince Danso-Abiam · Anthony Hanna · Alex Frieder
> **Prince's Role:** Feature Engineering, Validation & XGBoost/Gradient Boosting and streamlit deployment Lead

---

## Predictive Question

> *Can we predict whether a U.S. domestic flight will arrive 15 minutes or more late using airline, route, date, schedule, and congestion-related information from 2018, 2019, 2021, and 2022?*

**Target Variable:** `ArrDel15` - 1 = delayed ≥15 min, 0 = on time

---

## Repository Contents

| File | Description |
|---|---|
| `Fd_flight_delay_app.py` | Streamlit web application |
| `Fd_xgb_flight_delay_model.json` | Trained XGBoost model (version-independent JSON format) |
| `Fd_label_encoders.pkl` | Label encoders for categorical features |
| `Fd_feature_names.pkl` | Feature names in exact training order |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

---

## Dataset

| Property | Detail |
|---|---|
| **Source** | [Flight Status Prediction — Kaggle (robikscube)](https://www.kaggle.com/datasets/robikscube/flight-delay-dataset-20182022) |
| **Origin** | DOT Bureau of Transportation Statistics — Marketing Carrier On-Time Performance |
| **Years** | 2018, 2019, 2021, 2022 — 2020 excluded (COVID-19 disruption) |
| **Scope** | Top 6 busiest U.S. airports — ORD, ATL, DEN, DFW, CLT, LAX |
| **Final size** | ~4.1M rows · 16 features after cleaning |
| **Target balance** | 81% On Time · 19% Delayed |

### Connecting to the Dataset

The full cleaned dataset is hosted on Google Drive and can be loaded directly without downloading manually:

```python
import gdown
import pandas as pd

# Public Google Drive link — no login required
FILE_ID = "1Ja6U_fclo7Vg8FkXS0wCaDSz6W-z6iEJ"

gdown.download(
    f"https://drive.google.com/uc?id={FILE_ID}",
    "cleaned_flights.parquet",
    quiet=False
)

df = pd.read_parquet("cleaned_flights.parquet")
print(f"Loaded: {df.shape[0]:,} rows, {df.shape[1]} columns")
```

Alternatively, download directly from Kaggle using the API:

```python
import os, json, zipfile

# Configure Kaggle credentials
os.makedirs('/root/.kaggle', exist_ok=True)
with open('/root/.kaggle/kaggle.json', 'w') as f:
    json.dump({"username": "YOUR_USERNAME", "key": "YOUR_API_KEY"}, f)
os.chmod('/root/.kaggle/kaggle.json', 0o600)

files_to_download = [
    'Combined_Flights_2018.parquet',
    'Combined_Flights_2019.parquet',
    'Combined_Flights_2021.parquet',
    'Combined_Flights_2022.parquet',
]

for filename in files_to_download:
    os.system(f"kaggle datasets download -d robikscube/flight-delay-dataset-20182022 --file {filename} -p ./data/")
    zip_path = f'./data/{filename}.zip'
    if os.path.exists(zip_path):
        with zipfile.ZipFile(zip_path, 'r') as z:
            z.extractall('./data/')
        os.remove(zip_path)
```

---

## Feature Groups

| Group | Features | Count |
|---|---|---|
| **Temporal** | Year, Quarter, Month, DayofMonth, DayOfWeek | 5 |
| **Schedule** | CRSDepTime, CRSArrTime, DepTimeBlk | 3 |
| **Route** | Origin, Dest, Distance, DistanceGroup | 4 |
| **Airline** | Marketing_Airline_Network, Operating_Airline | 2 |
| **Congestion ✦** | origin_hourly_departures, dest_hourly_arrivals | 2 |

> ✦ **Engineered features** — derived entirely from scheduled flight data, completely leakage-free. Both outperformed Distance and DistanceGroup in SHAP importance.

**Leakage variables removed:** ArrDelay, ArrTime, AirTime, WheelsOn, TaxiIn, ActualElapsedTime, CRSElapsedTime (contained physically impossible negative values)

---

## Model

**Algorithm:** XGBoost (eXtreme Gradient Boosting)

**Justification:** Selected for its proven performance on large tabular classification tasks, sequential error-correcting tree architecture, native handling of class imbalance via `scale_pos_weight=4.25`, and built-in explainability through SHAP values. Supported by Kılıç & Sallan (2023) and Hatıpoğlu & Tosun (2024) who confirmed boosting models as top performers for U.S. flight delay prediction.

### Model Performance — All Three Models (Test Set)

| Model | Accuracy | Precision (Delayed) | Recall (Delayed) | F1 (Delayed) | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.57 | 0.26 | 0.62 | 0.37 | 0.6217 |
| **✦ XGBoost (Tuned)** | **0.69** | **0.33** | **0.64** | **0.44** | **0.7287** |
| Random Forest (Tuned) | 0.75 | 0.38 | 0.50 | 0.43 | 0.7186 |

**Primary metric: ROC-AUC** — evaluates model discrimination across all thresholds and is robust to the 19% class imbalance in the dataset.

**Overfitting check:** AUC gap (Train − Test) = 0.0037 → No overfitting 

**Tuning:** RandomizedSearchCV — 30 iterations, 3-fold stratified CV, T4 GPU acceleration

---

## Running the App Locally

### 1. Clone the repository

```bash
git clone https://github.com/pdansoabiam/pda_flight_predictor.git
cd pda_flight_predictor
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Ensure required model files are present

```
pda_flight_predictor/
├── Fd_flight_delay_app.py
├── Fd_xgb_flight_delay_model.json   ← required
├── Fd_label_encoders.pkl            ← required
├── Fd_feature_names.pkl             ← required
└── requirements.txt
```

### 4. Launch

```bash
streamlit run Fd_flight_delay_app.py
```

App opens at `http://localhost:8501`

---

## App Usage

1. Select flight **year, quarter, month, day**, and **day of week**
2. Choose **marketing airline**, **operating airline**, **origin** and **destination** airport
3. Enter **distance** (miles) and **distance group**
4. Enter **scheduled departure** and **arrival times** in hhmm format (e.g. 800 = 8:00 AM)
5. Select **departure time block**
6. Enter estimated **airport congestion** — number of flights at origin/destination in the same scheduled hour
7. Click **Predict**

The app returns:
-  **ON TIME** or  **DELAYED** prediction
- Delay probability percentage
- Decision threshold note (flights with ≥35% probability classified as delayed)

---

## Key Findings (SHAP)

1. **Time of day dominates** — `CRSDepTime` and `CRSArrTime` are the top predictors. Flights scheduled later in the day accumulate delays from earlier aircraft rotations.
2. **Seasonality is second** — `Month` confirms strong seasonal patterns. Summer and holiday periods drive the highest delay rates.
3. **Airline identity matters** — both marketing and operating carrier rank in the top 6, confirming consistent carrier-level performance differences.
4. **Congestion features validated** — engineered `origin_hourly_departures` and `dest_hourly_arrivals` outperformed `Distance` and `DistanceGroup` in SHAP importance — airport traffic volume at the scheduled hour adds real predictive signal.
5. **Geography matters less than timing** — `Origin` and `Dest` rank below time and airline features. *When* and *with whom* you fly matters more than *where*.

---

## Limitations

- Model trained on 2018–2022 data — may not reflect post-2022 airline patterns
- Extreme one-off events (e.g. Southwest December 2022 meltdown) are underweighted relative to their operational impact
- No external weather data — weather delay signal comes only from historical patterns embedded in temporal features
- One destination airport appears only in the test set — handled via neutral label encoding fallback
- Congestion features require access to the full day's scheduled flight data at inference time

---

## References

- Kılıç, M., & Sallan, J. M. (2023). Study of delay prediction in the US airport network. *Aerospace, 10*(4), 342. https://doi.org/10.3390/aerospace10040342
- Hatıpoğlu, B., & Tosun, Ö. (2024). Predictive modeling of flight delays using machine learning. *Applied Sciences, 14*(13), 5472. https://doi.org/10.3390/app14135472
- AlBassam, B. A., & AlShahrani, M. (2025). Flight delay prediction: Evaluating machine learning algorithms. *PLOS ONE, 20*(1), e0335141. https://doi.org/10.1371/journal.pone.0335141
- Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. *Proceedings of KDD 2016.*
- Lundberg, S., & Lee, S. I. (2017). A unified approach to interpreting model predictions. *NeurIPS 2017.*
- U.S. Bureau of Transportation Statistics. (2024). Marketing Carrier On-Time Performance. https://www.transtats.bts.gov

---

## 👤 Author

**Prince Danso-Abiam**
BSAN 6070 — Loyola Marymount University | Spring 2026
Role: Feature Engineering, Validation & XGBoost/Gradient Boosting Lead
