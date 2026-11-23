"""Seattle Collisions, Weather & Traffic - DSCI 510 project app.

How to run (from the repository root):
    streamlit run app/streamlit_app.py

All numbers in the app are calculated from our processed data when the app
runs. The forecast page also calls the Open-Meteo API live.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT / "cleaned_data"))
from collision_cleaning import RAW_COLLISIONS, load_collisions  # our shared cleaning function

st.set_page_config(page_title="Seattle Collision Explorer", page_icon="🚦", layout="wide")

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

# Same colors as the team's web dashboard (web/dashboard.css).
BLUE, ORANGE, RED, GRAY = "#9fd5c9", "#ffc670", "#ff8f94", "#8da5a6"
TEXT, MUTED, GRID = "#eef5f2", "#b9ceca", "rgba(215,231,227,0.10)"


# ---------------------------------------------------------------------------
# Look and feel
# ---------------------------------------------------------------------------
st.markdown("""
<style>
.block-container {padding-top: 2.2rem; max-width: 1350px;}
[data-testid="stMetric"] {background: #131b20; border: 1px solid rgba(215,231,227,.12);
                          border-radius: 14px; padding: 14px 18px;}
[data-testid="stMetricLabel"] p {color: #b9ceca; font-size: .8rem; text-transform: uppercase; letter-spacing: .06em;}
[data-testid="stMetricValue"] {font-size: 1.9rem; font-weight: 700;}
.kicker {color: #e1f5d2; font-size: .72rem; font-weight: 700; letter-spacing: .2em; text-transform: uppercase;}
.hero {padding: 26px 30px; margin-bottom: 18px; border: 1px solid rgba(215,231,227,.12); border-radius: 18px;
       background: linear-gradient(120deg, #18322c 0%, #121a1f 55%, #2a2219 100%);}
.hero h1 {margin: 6px 0 4px 0; font-size: 2.1rem; padding: 0;}
.hero p {color: #b9ceca; margin: 0; font-size: 1rem;}
.note {color: #b9ceca; font-size: .85rem;}
</style>
""", unsafe_allow_html=True)


def page_header(kicker, title, subtitle):
    st.markdown(f'<div class="hero"><div class="kicker">{kicker}</div><h1>{title}</h1><p>{subtitle}</p></div>',
                unsafe_allow_html=True)


def section(kicker, title):
    st.markdown(f'<div class="kicker">{kicker}</div>', unsafe_allow_html=True)
    st.subheader(title)


def show(fig, where=st, height=360):
    """Give every Plotly chart the same dark style, then draw it."""
    has_title = bool(fig.layout.title.text)
    fig.update_layout(height=height, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=MUTED, size=13),
                      # title at the top left, legend in its own row just above the plot
                      title=dict(x=0.01, xanchor="left", y=0.98, yanchor="top", font=dict(color=TEXT, size=16)),
                      margin=dict(l=20, r=20, t=90 if has_title else 50, b=20),
                      legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title=None,
                                  bgcolor="rgba(0,0,0,0)"))
    # automargin makes room for tick labels and axis titles so nothing gets cut off
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True)
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, automargin=True)
    where.plotly_chart(fig, width="stretch", theme=None)


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
    page_header("DSCI 510 · Course project", "Seattle Collisions, Weather & Traffic",
                "How do weather and traffic relate to car collisions in Seattle, and which collisions lead to injuries?")

    crashes = load_collisions_table()
    hours = load_hourly_table()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Collisions", f"{len(crashes):,}")
    c2.metric("Hours of weather data", f"{len(hours):,}")
    c3.metric("Injury rate", f"{crashes['injury'].mean():.0%}")
    c4.metric("Period (Sep – Jun)", "2023 – 2026")

    st.write("")
    section("Pipeline", "Where the data comes from")
    sources = [
        ("🌦️", "Open-Meteo API", "Hourly weather for Seattle", "Python API requests"),
        ("🚗", "SDOT Collisions", "Every reported collision", "Seattle open data export"),
        ("📊", "SDOT Traffic Volume", "Hourly counts from 7-day studies", "Socrata REST API"),
        ("🛣️", "Seattle Streets", "Street segments and intersections", "Seattle open data export"),
    ]
    for col, (icon, name, what, how) in zip(st.columns(4), sources):
        with col.container(border=True):
            st.markdown(f"### {icon}")
            st.markdown(f"**{name}**")
            st.markdown(f"<span class='note'>{what}<br>via {how}</span>", unsafe_allow_html=True)

    st.write("")
    section("Cleaning", "What we fixed in the collision data")
    raw = pd.read_csv(RAW_COLLISIONS, low_memory=False)
    clean = load_collisions(verbose=False)
    steps = pd.DataFrame({
        "Step": ["Raw export", "Rows from a different extract (removed)", "After cleaning",
                 "No time of day recorded (00:00)", "Severity not recorded yet", "Lat/lon swapped (fixed)"],
        "Records": [len(raw), len(raw) - len(clean), len(clean), int(crashes["time_unknown"].sum()),
                    int(crashes["injury"].isna().sum()), int((pd.to_numeric(crashes["LATITUDE"], errors="coerce") < 0).sum())],
    })
    left, right = st.columns([3, 2])
    left.dataframe(steps, hide_index=True, width="stretch",
                   column_config={"Records": st.column_config.NumberColumn(format="%d")})
    right.info("Reports without a time are stored as 00:00. We keep them in counts and maps, "
               "but leave them out of anything that uses the hour of the day.", icon="🕛")


def page_explore():
    page_header("Historical explorer", "Explore collisions",
                "Use the filters in the sidebar. Every chart and number updates from the cleaned data.")
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
    c1.metric("Collisions", f"{len(df):,}")
    c2.metric("Injury rate", f"{df['injury'].mean():.1%}")
    c3.metric("Pedestrian or cyclist", f"{((df['PEDCOUNT'] > 0) | (df['PEDCYLCOUNT'] > 0)).mean():.1%}")
    c4.metric("Fatalities", f"{int(df['FATALITIES'].sum()):,}")

    # Map, colored the same way as the web dashboard.
    points = df.dropna(subset=["lat", "lon"]).sample(min(len(df), 3000), random_state=510)
    points["outcome"] = np.select(
        [points["SEVERITYCODE"] >= 3, points["SEVERITYCODE"] == 2],
        ["Serious / fatal", "Injury"], default="Damage only / unknown")
    fig = px.scatter_map(points, lat="lat", lon="lon", color="outcome", hover_name="LOCATION",
                         hover_data={"COLLISIONTYPE": True, "lat": False, "lon": False},
                         color_discrete_map={"Serious / fatal": RED, "Injury": ORANGE, "Damage only / unknown": BLUE},
                         category_orders={"outcome": ["Damage only / unknown", "Injury", "Serious / fatal"]},
                         zoom=10.3, center={"lat": 47.61, "lon": -122.33}, map_style="carto-darkmatter",
                         title=f"Collision map ({len(points):,} shown)")
    fig.update_traces(marker={"size": 6, "opacity": 0.8})
    show(fig, height=560)

    left, right = st.columns(2)
    monthly = df.groupby(df["local_time"].dt.to_period("M")).size().reset_index(name="collisions")
    monthly["month"] = monthly["local_time"].dt.to_timestamp()
    fig = px.bar(monthly, x="month", y="collisions", title="Collisions per month", color_discrete_sequence=[BLUE])
    fig.update_layout(xaxis_title=None)
    with left.container(border=True):
        show(fig, height=330)

    known = df[~df["time_unknown"]]
    by_hour = known.groupby(known["local_time"].dt.hour).size().reindex(range(24), fill_value=0).reset_index()
    by_hour.columns = ["hour", "collisions"]
    fig = px.bar(by_hour, x="hour", y="collisions", title="By hour of day (time recorded only)",
                 color_discrete_sequence=[ORANGE])
    with right.container(border=True):
        show(fig, height=330)

    by_weather = (df[df["injury"].notna()].groupby("WEATHER")["injury"]
                  .agg(["count", "mean"]).query("count >= 20").sort_values("mean").reset_index())
    fig = px.bar(by_weather, x="mean", y="WEATHER", orientation="h", text_auto=".0%",
                 title="Injury rate by reported weather", labels={"mean": "injury rate", "WEATHER": ""},
                 color_discrete_sequence=[RED])
    fig.update_xaxes(tickformat=".0%")
    with st.container(border=True):
        show(fig, height=320)


def page_forecast():
    page_header("Machine learning", "Collision forecast",
                "Live weather from the Open-Meteo API goes into a gradient-boosting model trained on our hourly data.")
    model, auc, n_train, n_test = train_hourly_model()
    c1, c2, c3 = st.columns(3)
    c1.metric("Training hours", f"{n_train:,}")
    c2.metric("Test hours (later period)", f"{n_test:,}")
    c3.metric("Test AUC", f"{auc:.2f}", "0.5 = guessing", delta_color="off")

    st.write("")
    section("Live forecast", "Expected collisions in the coming hours")
    days = st.segmented_control("Forecast length", [1, 2, 3], default=2, format_func=lambda d: f"{d} day(s)") or 2
    try:
        forecast = get_forecast(days)
    except requests.RequestException as e:
        st.error(f"Could not reach the Open-Meteo API: {e}")
    else:
        forecast["expected collisions"] = model.predict(forecast[HOUR_FEATURES])
        forecast["time"] = forecast.index.tz_convert("America/Los_Angeles").tz_localize(None)
        fig = px.area(forecast, x="time", y="expected collisions", color_discrete_sequence=[BLUE],
                      hover_data={"rain": ":.2f", "temperature_2m": ":.0f"})
        rainy = forecast[forecast["rain"] > 0]
        fig.add_scatter(x=rainy["time"], y=rainy["expected collisions"], mode="markers", name="rain forecast",
                        marker={"color": ORANGE, "size": 8})
        fig.update_layout(xaxis_title=None)
        chart, numbers = st.columns([3, 1])
        with chart.container(border=True):
            show(fig, height=340)
        peak = forecast.loc[forecast["expected collisions"].idxmax()]
        numbers.metric("Expected in this period", f"{forecast['expected collisions'].sum():.0f}")
        numbers.metric("Highest-risk hour", peak["time"].strftime("%a %I %p"))
        numbers.metric("Hours with rain", f"{len(rainy)}")
    st.caption("Collisions are partly random, so treat this as a risk level, not an exact count.")

    st.divider()
    section("What if…?", "Try your own conditions")
    with st.container(border=True):
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

    scenario = pd.DataFrame([[hour, weekday, month, rain, rain * 2, since, temp, visibility * 5280, gusts, humidity]],
                            columns=HOUR_FEATURES)
    dry = scenario.assign(rain=0.0, rain_last_3h=0.0, hours_since_rain=48)
    expected = model.predict(scenario)[0]
    expected_dry = model.predict(dry)[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("Expected collisions this hour", f"{expected:.2f}")
    c2.metric("Chance of at least one", f"{1 - np.exp(-expected):.0%}")
    c3.metric("Compared with dry weather", f"{expected / expected_dry - 1:+.0%}")


def risk_gauge(prob, average):
    """Ring-style gauge, like the Prediction Lab in the web dashboard."""
    color = RED if prob > average + 0.05 else (BLUE if prob < average - 0.05 else ORANGE)
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=prob * 100, number={"suffix": "%", "font": {"size": 46, "color": TEXT}},
        gauge={"axis": {"range": [0, 100], "tickcolor": MUTED}, "bar": {"color": color, "thickness": 0.35},
               "bgcolor": "rgba(255,255,255,0.05)", "borderwidth": 0,
               "threshold": {"line": {"color": TEXT, "width": 3}, "thickness": 0.8, "value": average * 100}}))
    return fig


def page_injury():
    page_header("Prediction lab", "Injury prediction",
                "Describe a collision and a logistic regression estimates the chance that someone is injured.")
    model, scaler, columns, medians, scores, data = train_injury_model()

    def choices(col):
        return data[col].value_counts().index.tolist()

    inputs, result = st.columns([3, 2])
    with inputs.container(border=True):
        st.markdown("**Collision details**")
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
        with st.expander("Time and weather details"):
            crash["day_of_week"] = st.selectbox("Day of week", WEEKDAYS, index=4)
            crash["hour"] = st.slider("Hour", 0, 23, 17)
            crash["precipitation"] = st.slider("Rain (inches)", 0.0, 0.3, 0.0, 0.01)
            crash["temperature_2m"] = st.slider("Temperature (°F)", 20, 100, 50)

    X = make_injury_matrix(pd.DataFrame([crash]), columns).fillna(medians)
    prob = model.predict_proba(scaler.transform(X))[0, 1]
    average = data["injury"].mean()

    with result.container(border=True):
        st.markdown("**Chance someone is injured**")
        show(risk_gauge(prob, average), height=250)
        st.metric("Compared with an average collision", f"{(prob - average) * 100:+.0f} pts",
                  f"average = {average:.0%} (white line)", delta_color="off")

    c1, c2, c3 = st.columns(3)
    c1.metric("Test accuracy", f"{scores['accuracy']:.0%}", f"always 'no injury': {scores['baseline']:.0%}",
              delta_color="off")
    c2.metric("Test AUC", f"{scores['auc']:.2f}", "0.5 = guessing", delta_color="off")
    c3.metric("Tested on", f"{scores['test_size']:,}", "collisions after Sep 2025", delta_color="off")

    # Biggest coefficients = factors the model links most strongly to injury.
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
    top["direction"] = np.where(top["coefficient"] > 0, "more likely injury", "less likely injury")
    fig = px.bar(top, x="coefficient", y="feature", orientation="h", color="direction",
                 color_discrete_map={"more likely injury": RED, "less likely injury": BLUE},
                 title="Factors with the strongest link to injury (standardized coefficients)")
    fig.update_layout(yaxis_title=None)
    with st.container(border=True):
        show(fig, height=430)


# ---------------------------------------------------------------------------
# Navigation (tabs across the top of the page)
# ---------------------------------------------------------------------------
pages = [
    st.Page(page_overview, title="Overview", icon=":material/home:", default=True),
    st.Page(page_explore, title="Explore", icon=":material/map:"),
    st.Page(page_forecast, title="Forecast", icon=":material/insights:"),
    st.Page(page_injury, title="Injury prediction", icon=":material/medical_services:"),
]
st.navigation(pages, position="top").run()
