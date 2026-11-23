# Seattle Collisions, Weather & Traffic (Streamlit app)

Interactive Streamlit app for our DSCI 510 project.

| Page | What it does |
| --- | --- |
| Overview | Data sources and the cleaning steps, counted from the raw data |
| Explore | Filters, collision map, monthly and hourly charts, injury rate by weather |
| Forecast | Calls the Open-Meteo API live and predicts collisions for the coming hours (gradient boosting) |
| Injury prediction | Enter collision details and get the probability of an injury (logistic regression) |

## Run locally

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

## Data

- `data/Crashes-Analysis-Export-2023_09_20-2026_09_19.csv`: SDOT collision records (public).
- `cleaned_data/all_collisions_weather.csv`: the same collisions after cleaning (30 rows from a different extract removed), joined to hourly weather.
- `EDA/hourly_data.csv`: hourly Seattle weather from the Open-Meteo API.

Every number in the app is computed from these files when the app runs.
