# Metro Interstate Traffic Volume — Five-Model Forecasting and Clustering

A time-series machine-learning project using hourly traffic observations on Interstate 94 (westbound) between Minneapolis and St Paul, covering October 2012 to September 2018.

## Project overview

The notebook combines data cleaning, exploratory analysis, supervised forecasting, unsupervised clustering, statistical time-series modelling and deep learning. It evaluates five modelling families and compares forecast skill against seasonal-naive baselines.

## Dataset

Expected file: `Metro_Interstate_Traffic_Volume.csv`

The notebook reads this exact filename and identifies the source as the UCI Machine Learning Repository's **Metro Interstate Traffic Volume** dataset.

The notebook audits the raw data against 48,204 rows and 9 columns, handles repeated timestamps, impossible sensor values, missing hours and short gaps, then constructs an hourly time series.

The correct Metro Interstate Traffic Volume dataset has been included in this project folder as `Metro_Interstate_Traffic_Volume.csv`. The original upload was compressed as `.csv.gz`; it has been extracted and renamed to the exact filename expected by the notebook.

## Five modelling families

1. **Regression — LightGBM with Ridge baseline:** predicts next-hour traffic using calendar, weather and lag features; includes a day-ahead variant.
2. **Clustering — K-Means:** groups complete 24-hour daily traffic profiles by shape. This is an unsupervised analysis, not a forecast.
3. **Statistical — SARIMAX:** models daily seasonality (period 24), calendar effects and weather, with Holt-Winters as a baseline.
4. **Deep learning — LSTM (GRU alternative):** uses the previous 168 hours to forecast the next 24 hours.
5. **Transformer — PatchTST:** uses a patch-based time-series architecture for direct 24-hour forecasting.

## Data preparation and evaluation

- Audit raw data and identify duplicate timestamps, impossible values and missing observations.
- Aggregate observations to an hourly grid.
- Interpolate only short gaps (up to three hours); long gaps remain missing and split the series into continuous segments.
- Use chronological train/validation/test splits to preserve time order.
- Compare forecasting models with MAE, RMSE and sMAPE, alongside seasonal-naive baselines.
- Evaluate K-Means separately using cluster profiles and agreement with calendar day types.

## Selected findings recorded in the notebook

- LightGBM with lag features achieved approximately **141 vehicles/hour MAE** for the one-hour-ahead test task, about 58% better than the seasonal-naive baseline on that test set.
- K-Means recovered interpretable day-pattern groups corresponding broadly to weekdays, Saturdays, and Sundays/holidays.
- The SARIMAX model with a 24-hour seasonal period alone struggled to represent weekly structure; adding weekend-hour harmonics improved its results.
- LSTM, GRU and PatchTST achieved approximately 23–24% skill improvement over the seasonal-naive baseline for the notebook's 24-hour-ahead evaluation. Their small differences should not be treated as a reliable ranking from a single seed.

Forecast results from different test blocks are not directly interchangeable because the target windows and information available to each model differ. Use the notebook's comparison table and notes.

## Tools

Python, Pandas, NumPy, Matplotlib, scikit-learn, LightGBM, statsmodels, TensorFlow / Keras.

## Run the notebook

1. Keep `Metro_Interstate_Traffic_Volume.csv` in the same folder as the notebook (already included).
2. Install the libraries listed in the setup cell.
3. Run `Metro_Interstate_Traffic_Five_Models.ipynb` from top to bottom.
4. For a quick smoke test, set `fast_dev_run=True` in the configuration.

## Limitations

- Model families do not all receive the same future information: the notebook documents differences in weather/calendar inputs.
- SARIMAX is evaluated on a limited continuous test stretch.
- Neural model comparisons use a single seed and split; small differences are not statistically conclusive.
- Interpolated hours may be used as inputs but are not scored as targets.
- K-Means is descriptive clustering and does not produce forecast-error metrics.
