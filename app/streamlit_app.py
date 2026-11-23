"""Seattle Collisions, Weather & Traffic - DSCI 510 project app.

How to run (from the repository root):
    streamlit run app/streamlit_app.py

All numbers in the app are calculated from our processed data when the app
runs. The forecast page also calls the Open-Meteo API live.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
RAW_COLLISIONS = ROOT / "data/Crashes-Analysis-Export-2023_09_20-2026_09_19.csv"

st.set_page_config(page_title="Seattle Collisions", page_icon="🚦", layout="wide")

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Colors from Streamlit's default chart palette
BLUE, LIGHT_BLUE, ORANGE, RED = "#0068c9", "#83c9ff", "#ff8700", "#ff2b2b"


# ---------------------------------------------------------------------------
# Loading data
# ---------------------------------------------------------------------------
def time_not_recorded(incdttm):
    """Police reports without a time of day are stored as 00:00:00."""
    times = pd.to_datetime(incdttm.str[:19])
    return (times.dt.hour == 0) & (times.dt.minute == 0) & (times.dt.second == 0)


@st.cache_data
def load_collisions_table():
    df = pd.read_csv(ROOT / "cleaned_data/all_collisions_weather.csv", low_memory=False)
    df["local_time"] = pd.to_datetime(df["INCDTTM"].str[:19])
    df["year"] = df["local_time"].dt.year
    df["time_unknown"] = time_not_recorded(df["INCDTTM"])

    # 1 = someone was injured, 0 = property damage only, NaN = not recorded yet
    df["injury"] = np.where(df["SEVERITYCODE"].isna(), np.nan, (df["SEVERITYCODE"] > 1).astype(float))

    # Some rows have latitude and longitude swapped. Seattle: lat ~ 47, lon ~ -122.
    lat = pd.to_numeric(df["LATITUDE"], errors="coerce")
    lon = pd.to_numeric(df["LONGITUDE"], errors="coerce")
    swapped = lat < 0
    df["lat"] = np.where(swapped, lon, lat)
    df["lon"] = np.where(swapped, lat, lon)

    for col in ["COLLISIONTYPE", "JUNCTIONTYPE", "LIGHTCOND", "ROADCOND", "WEATHER"]:
        df[col] = df[col].fillna("Unknown")
    df["SPEEDING"] = df["SPEEDING"].fillna("N")
    return df


def add_rain_features(weather):
    """Open-Meteo precipitation stamped at hour t is the rain that fell during the
    hour BEFORE t, so we shift it back by one hour."""
    weather["rain"] = weather["precipitation"].shift(-1)
    weather["rain_last_3h"] = weather["rain"].rolling(3).sum()

    hours_since = []
    count = 0
    for amount in weather["rain"]:
        count = 0 if amount > 0 else count + 1
        hours_since.append(min(count, 336))     # cap at 2 weeks
    weather["hours_since_rain"] = hours_since
    return weather


def add_time_features(df):
    local = df.index.tz_convert("America/Los_Angeles")
    df["hour"] = local.hour
    df["weekday"] = local.dayofweek
    df["month"] = local.month
    return df


HOUR_FEATURES = ["hour", "weekday", "month", "rain", "rain_last_3h", "hours_since_rain",
                 "temperature_2m", "visibility", "wind_gusts_10m", "relative_humidity_2m"]


@st.cache_data
def load_hourly_table():
    """One row per hour: weather + number of collisions (with a recorded time)."""
    weather = pd.read_csv(ROOT / "EDA/hourly_data.csv")
    weather.index = pd.to_datetime(weather["date"], utc=True)
    weather = add_rain_features(weather)

    crashes = load_collisions_table()
    crashes = crashes[~crashes["time_unknown"]]
    crash_hours = pd.to_datetime(crashes["collision_time_utc"], utc=True).dt.floor("h")
    weather["collisions"] = crash_hours.value_counts().reindex(weather.index, fill_value=0)

    hours = add_time_features(weather)
    local = hours.index.tz_convert("America/Los_Angeles")
    keep = (local >= "2023-09-21") & (local < "2026-07-01")   # reports after Jun 2026 are incomplete
    keep &= ~((local.year == 2026) & (local.month == 2))      # Feb 2026 is a reporting gap
    return hours.loc[keep, HOUR_FEATURES + ["collisions"]].dropna()


def get_forecast(days):
    """Call the Open-Meteo forecast API and return an hourly DataFrame."""
    params = {
        "latitude": 47.6062, "longitude": -122.3321,
        "hourly": "precipitation,temperature_2m,visibility,wind_gusts_10m,relative_humidity_2m",
        "temperature_unit": "fahrenheit", "wind_speed_unit": "mph", "precipitation_unit": "inch",
        "timezone": "America/Los_Angeles", "past_days": 2, "forecast_days": days + 1,
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    weather = pd.DataFrame(data["hourly"])
    weather.index = pd.to_datetime(weather["time"]).dt.tz_localize(
        "America/Los_Angeles", ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    weather = weather[weather.index.notna()]
    weather = add_time_features(add_rain_features(weather))

    now = pd.Timestamp.now(tz="UTC").floor("h")
    upcoming = (weather.index >= now) & (weather.index < now + pd.Timedelta(days=days))
    return weather[upcoming].dropna(subset=HOUR_FEATURES)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------
@st.cache_resource
def train_hourly_model():
    """Gradient boosting with a Poisson loss: predicts collisions per hour."""
    hours = load_hourly_table()
    local = hours.index.tz_convert("America/Los_Angeles")
    train = hours[local < "2025-09-01"]
    test = hours[local >= "2025-09-01"]

    model = HistGradientBoostingRegressor(loss="poisson", max_leaf_nodes=15, learning_rate=0.05,
                                          max_iter=600, early_stopping=True, random_state=510)
    model.fit(train[HOUR_FEATURES], train["collisions"])
    expected = model.predict(test[HOUR_FEATURES])
    auc = roc_auc_score(test["collisions"] > 0, expected)

    model.fit(hours[HOUR_FEATURES], hours["collisions"])   # refit on everything for the app
    return model, auc, len(train), len(test)


INJURY_FEATURES = ["COLLISIONTYPE", "ADDRTYPE", "JUNCTIONTYPE", "LIGHTCOND", "ROADCOND", "WEATHER",
                   "SPEEDING", "HITPARKEDCAR", "day_of_week", "hour", "PEDCOUNT", "PEDCYLCOUNT",
                   "VEHCOUNT", "PERSONCOUNT", "precipitation", "temperature_2m"]


def make_injury_matrix(df, columns=None):
    """Same preprocessing as the team notebook: one-hot encoding with get_dummies."""
    X = pd.get_dummies(df[INJURY_FEATURES])
    if columns is not None:
        X = X.reindex(columns=columns, fill_value=0)
    return X


@st.cache_resource
def train_injury_model():
    """Logistic regression: team features + crash characteristics."""
    df = load_collisions_table()
    df = df[df["injury"].notna() & ~df["time_unknown"] & (df["local_time"] < "2026-07-01")]

    X = make_injury_matrix(df)
    medians = X.median()
    X = X.fillna(medians)
    y = df["injury"]

    is_train = df["local_time"] < "2025-09-01"
    scaler = StandardScaler().fit(X[is_train])
    model = LogisticRegression(max_iter=2000).fit(scaler.transform(X[is_train]), y[is_train])
    prob = model.predict_proba(scaler.transform(X[~is_train]))[:, 1]
    scores = {
        "accuracy": accuracy_score(y[~is_train], prob >= 0.5),
        "baseline": 1 - y[~is_train].mean(),      # always guess "no injury"
        "auc": roc_auc_score(y[~is_train], prob),
        "test_size": int((~is_train).sum()),
    }
    return model, scaler, list(X.columns), medians, scores, df


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
def page_overview():
    st.title("🚦 Seattle Collisions, Weather & Traffic")
    st.caption("DSCI 510 course project · SDOT collision records, Sep 2023 – Jun 2026")
    st.write("How do weather and traffic relate to car collisions in Seattle, and which collisions "
             "lead to injuries? Use the pages in the sidebar to explore the data and try our models.")

    crashes = load_collisions_table()
    hours = load_hourly_table()
    c1, c2, c3 = st.columns(3)
    c1.metric("Collisions", f"{len(crashes):,}", border=True)
    c2.metric("Hours of weather data", f"{len(hours):,}", border=True)
    c3.metric("Collisions with an injury", f"{crashes['injury'].mean():.0%}", border=True)

    st.subheader("Data sources")
    st.dataframe(pd.DataFrame({
        "Source": ["Open-Meteo", "SDOT Collisions", "SDOT Traffic Volume", "Seattle Streets"],
        "What it contains": ["Hourly weather for Seattle", "Every reported collision",
                             "Hourly vehicle counts from 7-day studies", "Street segments and intersections"],
        "How we got it": ["Python API requests", "Seattle open data export", "Socrata REST API",
                          "Seattle open data export"],
    }), hide_index=True, width="stretch")

    st.subheader("Data cleaning")
    raw = pd.read_csv(RAW_COLLISIONS, low_memory=False)
    # rows from a different extract spell out ADDRTYPE ("Block"/"Intersection"); see EDA_with_cleaning_fixes.ipynb
    other_extract = int(raw["ADDRTYPE"].isin(["Block", "Intersection"]).sum())
    st.dataframe(pd.DataFrame({
        "Step": ["Raw export", "Rows from a different extract (removed)", "After cleaning",
                 "No time of day recorded (00:00)", "Severity not recorded yet", "Latitude/longitude swapped (fixed)"],
        "Records": [len(raw), other_extract, len(raw) - other_extract, int(crashes["time_unknown"].sum()),
                    int(crashes["injury"].isna().sum()),
                    int((pd.to_numeric(crashes["LATITUDE"], errors="coerce") < 0).sum())],
    }), hide_index=True, width="stretch",
        column_config={"Records": st.column_config.NumberColumn(format="localized")})
    st.caption("Reports without a time are stored as 00:00. We keep them in the counts and on the map, "
               "but leave them out of anything that uses the hour of the day.")


def page_explore():
    st.title("Explore collisions")
    st.caption("Use the filters in the sidebar. All numbers and charts update with the filters.")
    df = load_collisions_table()

    with st.sidebar:
        st.header("Filters")
        years = st.slider("Years", 2023, 2026, (2023, 2026))
        weather = st.multiselect("Weather", sorted(df["WEATHER"].unique()))
        road = st.multiselect("Road condition", sorted(df["ROADCOND"].unique()))
        where = st.multiselect("Location type", ["Block", "Intersection"])

    df = df[df["year"].between(years[0], years[1])]
    if weather:
        df = df[df["WEATHER"].isin(weather)]
    if road:
        df = df[df["ROADCOND"].isin(road)]
    if where:
        df = df[df["ADDRTYPE"].isin(where)]
    if df.empty:
        st.warning("No collisions match these filters.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Collisions", f"{len(df):,}", border=True)
    c2.metric("Injury rate", f"{df['injury'].mean():.1%}", border=True)
    c3.metric("Pedestrian or cyclist involved", f"{((df['PEDCOUNT'] > 0) | (df['PEDCYLCOUNT'] > 0)).mean():.1%}",
              border=True)
    c4.metric("Fatalities", f"{int(df['FATALITIES'].sum()):,}", border=True)

    tab_map, tab_time, tab_weather = st.tabs(["Map", "Over time", "Weather"])

    with tab_map:
        points = df.dropna(subset=["lat", "lon"]).sample(min(len(df), 3000), random_state=510)
        points["outcome"] = np.select(
            [points["SEVERITYCODE"] >= 3, points["SEVERITYCODE"] == 2],
            ["Serious injury or fatality", "Injury"], default="No injury / unknown")
        fig = px.scatter_map(points, lat="lat", lon="lon", color="outcome", hover_name="LOCATION",
                             hover_data={"COLLISIONTYPE": True, "lat": False, "lon": False},
                             color_discrete_map={"No injury / unknown": LIGHT_BLUE, "Injury": ORANGE,
                                                 "Serious injury or fatality": RED},
                             category_orders={"outcome": ["No injury / unknown", "Injury", "Serious injury or fatality"]},
                             zoom=10.3, center={"lat": 47.61, "lon": -122.33}, map_style="carto-positron", height=560)
        fig.update_traces(marker={"size": 6, "opacity": 0.8})
        fig.update_layout(legend_title_text=None, margin=dict(l=0, r=0, t=0, b=0))
        st.plotly_chart(fig, width="stretch")
        st.caption(f"Showing a random sample of {len(points):,} collisions.")

    with tab_time:
        left, right = st.columns(2)
        monthly = df.groupby(df["local_time"].dt.to_period("M")).size().reset_index(name="collisions")
        monthly["month"] = monthly["local_time"].dt.to_timestamp()
        left.markdown("**Collisions per month**")
        fig = px.bar(monthly, x="month", y="collisions", color_discrete_sequence=[BLUE])
        fig.update_layout(xaxis_title=None)
        left.plotly_chart(fig, width="stretch")

        known = df[~df["time_unknown"]]
        by_hour = known.groupby(known["local_time"].dt.hour).size().reindex(range(24), fill_value=0).reset_index()
        by_hour.columns = ["hour", "collisions"]
        right.markdown("**Collisions by hour of day** (reports with a recorded time)")
        fig = px.bar(by_hour, x="hour", y="collisions", color_discrete_sequence=[BLUE])
        right.plotly_chart(fig, width="stretch")

    with tab_weather:
        by_weather = (df[df["injury"].notna()].groupby("WEATHER")["injury"]
                      .agg(["count", "mean"]).query("count >= 20").sort_values("mean").reset_index())
        st.markdown("**Share of collisions with an injury, by reported weather**")
        fig = px.bar(by_weather, x="mean", y="WEATHER", orientation="h", text_auto=".0%",
                     labels={"mean": "injury rate", "WEATHER": ""}, color_discrete_sequence=[BLUE])
        fig.update_xaxes(tickformat=".0%")
        st.plotly_chart(fig, width="stretch")
        st.caption("Only weather types with at least 20 collisions are shown.")


def page_forecast():
    st.title("Collision forecast")
    model, auc, n_train, n_test = train_hourly_model()
    st.caption(f"Gradient boosting model trained on {n_train:,} hours and tested on {n_test:,} later hours "
               f"(test AUC {auc:.2f}; 0.5 would be random guessing). Collisions are partly random, "
               "so treat the numbers as a risk level, not an exact count.")

    tab_live, tab_whatif = st.tabs(["Next few days", "What if"])

    with tab_live:
        days = st.radio("Forecast length", [1, 2, 3], index=1, horizontal=True,
                        format_func=lambda d: f"{d} day" if d == 1 else f"{d} days")
        try:
            forecast = get_forecast(days)
        except requests.RequestException as e:
            st.error(f"Could not reach the Open-Meteo weather API. Please try again in a minute. ({e})")
        else:
            forecast["expected collisions"] = model.predict(forecast[HOUR_FEATURES])
            forecast["time"] = forecast.index.tz_convert("America/Los_Angeles").tz_localize(None)
            rainy = forecast[forecast["rain"] > 0]
            peak = forecast.loc[forecast["expected collisions"].idxmax()]

            c1, c2, c3 = st.columns(3)
            c1.metric("Expected collisions", f"{forecast['expected collisions'].sum():.0f}", border=True)
            c2.metric("Highest-risk hour", peak["time"].strftime("%a %-I %p"), border=True)
            c3.metric("Hours with rain in the forecast", f"{len(rainy)}", border=True)

            fig = px.line(forecast, x="time", y="expected collisions", color_discrete_sequence=[BLUE],
                          hover_data={"rain": ":.2f", "temperature_2m": ":.0f"})
            fig.add_scatter(x=rainy["time"], y=rainy["expected collisions"], mode="markers", name="rain forecast",
                            marker={"color": ORANGE, "size": 7})
            fig.update_layout(xaxis_title=None, yaxis_title="expected collisions per hour")
            st.plotly_chart(fig, width="stretch")
            st.caption("Weather forecast from the Open-Meteo API, loaded when you open this page.")

    with tab_whatif:
        st.write("Change the conditions to see how the model's estimate for one hour changes.")
        c1, c2, c3 = st.columns(3)
        hour = c1.slider("Hour of day", 0, 23, 17)
        weekday = c1.selectbox("Day of week", range(7), index=4, format_func=lambda i: WEEKDAYS[i])
        month = c1.selectbox("Month", range(1, 13), index=10, format_func=lambda i: MONTHS[i - 1])
        rain = c2.slider("Rain this hour (inches)", 0.0, 0.3, 0.0, 0.01)
        since = 0 if rain > 0 else c2.slider("Hours since it last rained", 0, 72, 24)
        temp = c2.slider("Temperature (°F)", 20, 100, 50)
        visibility = c3.slider("Visibility (miles)", 0.5, 15.0, 10.0, 0.5)
        gusts = c3.slider("Wind gusts (mph)", 0, 50, 10)
        humidity = c3.slider("Humidity (%)", 20, 100, 75)

        scenario = pd.DataFrame([[hour, weekday, month, rain, rain * 2, since, temp, visibility * 5280, gusts,
                                  humidity]], columns=HOUR_FEATURES)
        dry = scenario.assign(rain=0.0, rain_last_3h=0.0, hours_since_rain=48)
        expected = model.predict(scenario)[0]
        expected_dry = model.predict(dry)[0]
        st.divider()
        c1, c2, c3 = st.columns(3)
        c1.metric("Expected collisions this hour", f"{expected:.2f}", border=True)
        c2.metric("Chance of at least one", f"{1 - np.exp(-expected):.0%}", border=True)
        c3.metric("Compared with dry weather", f"{expected / expected_dry - 1:+.0%}", border=True)


def page_injury():
    st.title("Injury prediction")
    model, scaler, columns, medians, scores, data = train_injury_model()
    st.caption(f"Logistic regression tested on {scores['test_size']:,} collisions after Sep 2025: "
               f"accuracy {scores['accuracy']:.0%} (always guessing 'no injury' gets {scores['baseline']:.0%}), "
               f"AUC {scores['auc']:.2f}.")

    def choices(col):
        return data[col].value_counts().index.tolist()

    left, right = st.columns([2, 1], gap="large")
    with left:
        st.subheader("Collision details")
        c1, c2, c3 = st.columns(3)
        crash = {
            "COLLISIONTYPE": c1.selectbox("Collision type", choices("COLLISIONTYPE")),
            "ADDRTYPE": c1.selectbox("Location", ["Block", "Intersection"]),
            "JUNCTIONTYPE": c1.selectbox("Junction", choices("JUNCTIONTYPE")),
            "PEDCOUNT": c2.number_input("Pedestrians", 0, 5, 0),
            "PEDCYLCOUNT": c2.number_input("Cyclists", 0, 5, 0),
            "VEHCOUNT": c2.number_input("Vehicles", 0, 10, 2),
            "PERSONCOUNT": c2.number_input("People involved", 0, 20, 2),
            "LIGHTCOND": c3.selectbox("Light", choices("LIGHTCOND")),
            "ROADCOND": c3.selectbox("Road surface", choices("ROADCOND")),
            "WEATHER": c3.selectbox("Weather", choices("WEATHER")),
            "SPEEDING": "Y" if c3.checkbox("Speeding involved") else "N",
            "HITPARKEDCAR": "Y" if c3.checkbox("Hit a parked car") else "N",
            "day_of_week": "Friday", "hour": 17, "precipitation": 0.0, "temperature_2m": 50.0,
        }
        with st.expander("Time and weather"):
            crash["day_of_week"] = st.selectbox("Day of week", WEEKDAYS, index=4)
            crash["hour"] = st.slider("Hour", 0, 23, 17)
            crash["precipitation"] = st.slider("Rain (inches)", 0.0, 0.3, 0.0, 0.01)
            crash["temperature_2m"] = st.slider("Temperature (°F)", 20, 100, 50)

    X = make_injury_matrix(pd.DataFrame([crash]), columns).fillna(medians)
    prob = model.predict_proba(scaler.transform(X))[0, 1]
    average = data["injury"].mean()

    with right:
        st.subheader("Result")
        st.metric("Chance someone is injured", f"{prob:.0%}",
                  f"{(prob - average) * 100:+.0f} points vs. average", delta_color="inverse", border=True)
        st.progress(float(prob))
        st.caption(f"An average collision in our data has a {average:.0%} chance of an injury.")

    st.subheader("Factors with the strongest link to injury")
    coefs = pd.Series(model.coef_[0], index=columns).sort_values()
    top = pd.concat([coefs.head(6), coefs.tail(6)]).reset_index()
    top.columns = ["feature", "coefficient"]
    nice = {"COLLISIONTYPE_": "Collision type: ", "JUNCTIONTYPE_": "Junction: ", "LIGHTCOND_": "Light: ",
            "ROADCOND_": "Road: ", "WEATHER_": "Weather: ", "ADDRTYPE_": "Location: ", "SPEEDING_": "Speeding: ",
            "HITPARKEDCAR_": "Hit parked car: ", "day_of_week_": "Day: ", "PEDCOUNT": "Pedestrians",
            "PEDCYLCOUNT": "Cyclists", "VEHCOUNT": "Vehicles", "PERSONCOUNT": "People involved",
            "precipitation": "Rain", "temperature_2m": "Temperature", "hour": "Hour"}
    for old, new in nice.items():
        top["feature"] = top["feature"].str.replace(old, new, regex=False)
    top["feature"] = top["feature"].str.slice(0, 45)
    top["direction"] = np.where(top["coefficient"] > 0, "More likely injury", "Less likely injury")
    fig = px.bar(top, x="coefficient", y="feature", orientation="h", color="direction",
                 color_discrete_map={"More likely injury": RED, "Less likely injury": BLUE}, height=430)
    fig.update_layout(yaxis_title=None, xaxis_title="standardized coefficient", legend_title_text=None)
    st.plotly_chart(fig, width="stretch")


# ---------------------------------------------------------------------------
# Navigation (sidebar)
# ---------------------------------------------------------------------------
pages = [
    st.Page(page_overview, title="Overview", icon="🏠", default=True),
    st.Page(page_explore, title="Explore", icon="🗺️"),
    st.Page(page_forecast, title="Forecast", icon="📈"),
    st.Page(page_injury, title="Injury prediction", icon="🚑"),
]
st.navigation(pages).run()
