from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ============================================================
# File paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
MULTI_ENG_DIR = BASE_DIR.parent
BEN_DIR = MULTI_ENG_DIR / "ben"

TRAFFIC_FILE = BASE_DIR / "tfnsw_api" / "tfnsw_hourly_traffic.csv"


def find_file(folder, pattern):
    matches = list(folder.glob(pattern))

    if len(matches) == 0:
        raise FileNotFoundError(
            f"No file found in {folder} matching pattern: {pattern}"
        )

    return matches[0]


ALLIANZ_FILE = find_file(BEN_DIR, "*allianz*.csv")
HORDERN_FILE = find_file(BEN_DIR, "*hordern*.csv")
RANDWICK_FILE = find_file(BEN_DIR, "*randwick*.csv")

print("BASE_DIR:", BASE_DIR)
print("BEN_DIR:", BEN_DIR)
print("Traffic file:", TRAFFIC_FILE)
print("Traffic exists:", TRAFFIC_FILE.exists())
print("Allianz file:", ALLIANZ_FILE)
print("Hordern file:", HORDERN_FILE)
print("Randwick file:", RANDWICK_FILE)


# ============================================================
# Load traffic data
# ============================================================

traffic = pd.read_csv(TRAFFIC_FILE)

traffic["date"] = pd.to_datetime(traffic["date"]).dt.tz_localize(None).dt.normalize()

# ============================================================
# Load event data
# ============================================================

allianz = pd.read_csv(ALLIANZ_FILE)
hordern = pd.read_csv(HORDERN_FILE)
randwick = pd.read_csv(RANDWICK_FILE)


def standardise_event_file(event_df):
    """
    Makes event CSVs use the same column names.
    Required final columns: date, estimated_attendance
    """

    event_df = event_df.copy()

    if "attendance" in event_df.columns and "estimated_attendance" not in event_df.columns:
        event_df = event_df.rename(columns={"attendance": "estimated_attendance"})

    if "Date" in event_df.columns and "date" not in event_df.columns:
        event_df = event_df.rename(columns={"Date": "date"})

    if "Estimated Attendance" in event_df.columns and "estimated_attendance" not in event_df.columns:
        event_df = event_df.rename(columns={"Estimated Attendance": "estimated_attendance"})

    if "estimated_attendance" not in event_df.columns:
        event_df["estimated_attendance"] = 0

    event_df["date"] = pd.to_datetime(event_df["date"], errors="coerce").dt.normalize()
    event_df["estimated_attendance"] = pd.to_numeric(
        event_df["estimated_attendance"],
        errors="coerce"
    ).fillna(0)

    event_df = event_df.dropna(subset=["date"])

    return event_df[["date", "estimated_attendance"]]


allianz = standardise_event_file(allianz)
hordern = standardise_event_file(hordern)
randwick = standardise_event_file(randwick)

events = pd.concat(
    [allianz, hordern, randwick],
    ignore_index=True
)

events_daily = events.groupby("date").agg(
    event_flag=("date", "count"),
    max_attendance=("estimated_attendance", "max"),
    total_attendance=("estimated_attendance", "sum")
).reset_index()

events_daily["max_attendance"] = events_daily["max_attendance"].fillna(0)
events_daily["total_attendance"] = events_daily["total_attendance"].fillna(0)


# ============================================================
# Merge traffic and event data
# ============================================================

df = traffic.merge(events_daily, on="date", how="left")

df["event_flag"] = df["event_flag"].fillna(0)
df["max_attendance"] = df["max_attendance"].fillna(0)
df["total_attendance"] = df["total_attendance"].fillna(0)


# ============================================================
# Feature engineering
# ============================================================

df["day_of_year"] = df["date"].dt.dayofyear

# If day_of_week is not already in the traffic file, create it.
# Monday = 1, Sunday = 7
if "day_of_week" not in df.columns:
    df["day_of_week"] = df["date"].dt.dayofweek + 1

if "month" not in df.columns:
    df["month"] = df["date"].dt.month

if "year" not in df.columns:
    df["year"] = df["date"].dt.year

if "public_holiday" not in df.columns:
    df["public_holiday"] = 0

if "school_holiday" not in df.columns:
    df["school_holiday"] = 0

df["is_weekend"] = (df["day_of_week"] >= 6).astype(int)
df["public_holiday"] = df["public_holiday"].astype(int)
df["school_holiday"] = df["school_holiday"].astype(int)
df["has_event"] = (df["event_flag"] > 0).astype(int)

FEATURES = [
    "month",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "public_holiday",
    "school_holiday",
    "has_event",
    "max_attendance",
    "total_attendance",
]

TARGET = "daily_total"

# Make sure model columns are numeric
for col in FEATURES + [TARGET]:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=FEATURES + [TARGET])


# ============================================================
# Train/test split
# ============================================================

train = df[(df["year"] >= 2017) & (df["year"] <= 2018)].copy()
test = df[df["year"] == 2019].copy()

if len(train) == 0 or len(test) == 0:
    raise ValueError(
        "Train or test data is empty. Check that your traffic CSV contains 2017, 2018 and 2019 rows."
    )

X_train = train[FEATURES]
y_train = train[TARGET]

X_test = test[FEATURES]
y_test = test[TARGET]


# ============================================================
# Train Random Forest
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)

model.fit(X_train, y_train)

test["predicted_traffic"] = model.predict(X_test)


# ============================================================
# Create historic average for comparison
# ============================================================

historic_lookup = train.groupby(["month", "day_of_week"])[TARGET].mean().reset_index()
historic_lookup = historic_lookup.rename(columns={TARGET: "historic_average"})

test = test.merge(
    historic_lookup,
    on=["month", "day_of_week"],
    how="left"
)

overall_average = train[TARGET].mean()
test["historic_average"] = test["historic_average"].fillna(overall_average)


# ============================================================
# Metrics
# ============================================================

rmse = np.sqrt(mean_squared_error(y_test, test["predicted_traffic"]))
mae = mean_absolute_error(y_test, test["predicted_traffic"])
r2 = r2_score(y_test, test["predicted_traffic"])

print("\nModel Performance")
print(f"Training rows: {len(train):,}")
print(f"Test rows    : {len(test):,}")
print(f"RMSE: {rmse:.0f} vehicles/day")
print(f"MAE : {mae:.0f} vehicles/day")
print(f"R²  : {r2:.3f}")


# ============================================================
# Future prediction + display logic
# ============================================================

def congestion_level(value):
    if value < 10000:
        return "Low"
    if value < 13000:
        return "Moderate"
    if value < 15000:
        return "High"
    return "Severe"


def make_prediction_row(date_str, event_name="No event", has_event=0,
                        max_attendance=0, total_attendance=0,
                        public_holiday=0, school_holiday=0):
    date = pd.Timestamp(date_str).normalize()

    return {
        "date": date,
        "date_display": date.strftime("%Y-%m-%d"),
        "event_name": event_name,

        "month": date.month,
        "day_of_week": date.dayofweek + 1,
        "day_of_year": date.dayofyear,
        "is_weekend": 1 if date.dayofweek >= 5 else 0,
        "public_holiday": public_holiday,
        "school_holiday": school_holiday,
        "has_event": has_event,
        "max_attendance": max_attendance,
        "total_attendance": total_attendance,
    }


# ============================================================
# Load Ben's future event file
# ============================================================

FUTURE_EVENTS_FILE = BEN_DIR / "events_cleaned.csv"

future_rows = []

if FUTURE_EVENTS_FILE.exists():
    future_events = pd.read_csv(FUTURE_EVENTS_FILE)

    future_events["date"] = (
        pd.to_datetime(future_events["start"], errors="coerce")
        .dt.tz_localize(None)
        .dt.normalize()
    )

    # Choose the future dates you want to display
    dates_to_predict = [
        "2026-05-22",
        "2026-05-23",
        "2026-06-02",
        "2027-10-13",
        "2027-09-14",
    ]

    for date_str in dates_to_predict:
        date = pd.Timestamp(date_str).normalize()
        events_that_day = future_events[future_events["date"] == date]

        if len(events_that_day) > 0:
            has_event = 1

            # Your main.py uses intensity * 50000
            max_attendance = events_that_day["intensity"].max() * 50000
            total_attendance = max_attendance * len(events_that_day)

            event_name = "; ".join(events_that_day["summary"].astype(str).tolist())
        else:
            has_event = 0
            max_attendance = 0
            total_attendance = 0
            event_name = "No event"

        future_rows.append(
            make_prediction_row(
                date_str=date_str,
                event_name=event_name,
                has_event=has_event,
                max_attendance=max_attendance,
                total_attendance=total_attendance,
                public_holiday=0,
                school_holiday=0,
            )
        )

else:
    print(f"\nWarning: could not find {FUTURE_EVENTS_FILE}")
    print("Using manual future scenarios instead.")

    future_rows = [
        make_prediction_row(
            "2027-09-13",
            event_name="Manual event scenario",
            has_event=1,
            max_attendance=45000,
            total_attendance=45000,
        ),
        make_prediction_row(
            "2027-09-14",
            event_name="No event",
            has_event=0,
            max_attendance=0,
            total_attendance=0,
        ),
    ]


future_df = pd.DataFrame(future_rows)

# Predict future traffic using the trained Random Forest
future_df["predicted_traffic"] = model.predict(future_df[FEATURES])


# ============================================================
# Historic average comparison
# ============================================================

historic_lookup = train.groupby(["month", "day_of_week"])[TARGET].mean().reset_index()
historic_lookup = historic_lookup.rename(columns={TARGET: "historic_average"})

future_df = future_df.merge(
    historic_lookup,
    on=["month", "day_of_week"],
    how="left"
)

overall_average = train[TARGET].mean()
future_df["historic_average"] = future_df["historic_average"].fillna(overall_average)


# ============================================================
# Assistance calculations
# ============================================================

# Data-driven threshold:
# 80th percentile = traffic level higher than 80% of historical training days
ASSISTANCE_THRESHOLD = train[TARGET].quantile(0.80)

print(f"\nAssistance threshold: {ASSISTANCE_THRESHOLD:,.0f} vehicles/day")

future_df["difference"] = future_df["predicted_traffic"] - future_df["historic_average"]
future_df["percent_change"] = (future_df["difference"] / future_df["historic_average"]) * 100

future_df["vehicles_above_threshold"] = future_df["predicted_traffic"] - ASSISTANCE_THRESHOLD
future_df["assistance_needed"] = future_df["predicted_traffic"] > ASSISTANCE_THRESHOLD

future_df["congestion"] = future_df["predicted_traffic"].apply(congestion_level)

display_df = future_df.copy()


# ============================================================
# Terminal output
# ============================================================

print("\nFuture Traffic Assistance Display Data")
print(
    display_df[
        [
            "date_display",
            "event_name",
            "historic_average",
            "predicted_traffic",
            "difference",
            "percent_change",
            "congestion",
            "assistance_needed",
        ]
    ].to_string(index=False)
)


# ============================================================
# Matplotlib number-style display
# ============================================================

fig, ax = plt.subplots(figsize=(13, 7))
ax.axis("off")

fig.suptitle(
    "Future Traffic Forecast Assistance Display",
    fontsize=18,
    fontweight="bold"
)

station_text = "Station 55432 — Cleveland Street, West of Anzac Parade"

ax.text(
    0.5,
    0.93,
    station_text,
    ha="center",
    va="center",
    fontsize=12
)

# Main card uses the first future prediction
latest = display_df.iloc[0]
assistance_text = "YES" if latest["assistance_needed"] else "NO"

summary_text = (
    f"Selected Date: {latest['date_display']}\n"
    f"Scenario: {latest['event_name']}\n\n"
    f"Historic Average: {latest['historic_average']:,.0f} vehicles/day\n"
    f"Predicted Traffic: {latest['predicted_traffic']:,.0f} vehicles/day\n"
    f"Difference: {latest['difference']:+,.0f} vehicles/day\n"
    f"Change: {latest['percent_change']:+.1f}%\n"
    f"Congestion Level: {latest['congestion']}\n"
    f"Assistance Needed: {assistance_text}"
)

ax.text(
    0.5,
    0.66,
    summary_text,
    ha="center",
    va="center",
    fontsize=14,
    bbox=dict(
        boxstyle="round,pad=0.8",
        edgecolor="black",
        facecolor="white"
    )
)

table_data = []

for _, row in display_df.iterrows():
    table_data.append([
        row["date_display"],
        str(row["event_name"])[:28],
        f"{row['historic_average']:,.0f}",
        f"{row['predicted_traffic']:,.0f}",
        f"{row['difference']:+,.0f}",
        f"{row['percent_change']:+.1f}%",
        row["congestion"],
        "YES" if row["assistance_needed"] else "NO",
    ])

column_labels = [
    "Date",
    "Event",
    "Historic Avg",
    "Predicted",
    "Difference",
    "% Change",
    "Congestion",
    "Assist?"
]

table = ax.table(
    cellText=table_data,
    colLabels=column_labels,
    loc="lower center",
    cellLoc="center",
    colLoc="center",
    bbox=[0.01, 0.04, 0.98, 0.36]
)

table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.5)

plt.tight_layout()

output_file = BASE_DIR / "future_traffic_forecast_display.png"
plt.savefig(output_file, dpi=300)

print(f"\nSaved display to: {output_file}")

plt.show()