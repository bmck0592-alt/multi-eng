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
PROJECT_DIR = BASE_DIR.parent
PROJECT_PARENT = PROJECT_DIR.parent

SEARCH_DIRS = [
    BASE_DIR,
    BASE_DIR / "tfnsw_api",
    PROJECT_DIR,
    PROJECT_DIR / "tfnsw_api",
    PROJECT_DIR / "ben",
    PROJECT_PARENT,
    PROJECT_PARENT / "ben",
]


def find_file(pattern, folders=SEARCH_DIRS):
    """Find the first file matching a pattern in likely project folders."""
    for folder in folders:
        if folder.exists():
            matches = sorted(folder.glob(pattern))
            if matches:
                return matches[0]

    # Final fallback: recursive search from the project area.
    for root in [BASE_DIR, PROJECT_DIR, PROJECT_PARENT]:
        if root.exists():
            matches = sorted(root.rglob(pattern))
            if matches:
                return matches[0]

    raise FileNotFoundError(f"No file found matching pattern: {pattern}")


TRAFFIC_FILE = find_file("tfnsw_hourly_traffic.csv")
SCG_FILE = find_file("*scg*events*clean*.csv")
HORDERN_FILE = find_file("*hordern*events*clean*.csv")
RANDWICK_FILE = find_file("*randwick*events*clean*.csv")
ALLIANZ_FILE = find_file("*allianz*sydney*football*stadium*events*clean*.csv")

print("BASE_DIR:", BASE_DIR)
print("Traffic file:", TRAFFIC_FILE)
print("SCG file:", SCG_FILE)
print("Hordern file:", HORDERN_FILE)
print("Randwick file:", RANDWICK_FILE)
print("Allianz file:", ALLIANZ_FILE)


# ============================================================
# Load traffic data and reshape to hourly
# ============================================================

traffic = pd.read_csv(TRAFFIC_FILE)
traffic["date"] = pd.to_datetime(traffic["date"]).dt.tz_localize(None).dt.normalize()

hour_cols = [f"hour_{str(i).zfill(2)}" for i in range(24)]
missing_hour_cols = [col for col in hour_cols if col not in traffic.columns]
if missing_hour_cols:
    raise ValueError(f"Traffic file is missing hourly columns: {missing_hour_cols}")

hourly = traffic.melt(
    id_vars=["date", "year", "month", "day_of_week", "public_holiday", "school_holiday"],
    value_vars=hour_cols,
    var_name="hour_col",
    value_name="vehicle_count",
)

hourly["hour"] = hourly["hour_col"].str.extract(r"(\d+)").astype(int)
hourly = hourly.drop(columns="hour_col")
hourly = hourly.dropna(subset=["vehicle_count"])
hourly = hourly.sort_values(["date", "hour"]).reset_index(drop=True)


# ============================================================
# Load and standardise event data
# ============================================================

def load_event_file(path, default_attendance, venue_type, hour_column=None):
    event_df = pd.read_csv(path).copy()

    if "attendance" in event_df.columns and "estimated_attendance" not in event_df.columns:
        event_df = event_df.rename(columns={"attendance": "estimated_attendance"})

    if "Estimated Attendance" in event_df.columns and "estimated_attendance" not in event_df.columns:
        event_df = event_df.rename(columns={"Estimated Attendance": "estimated_attendance"})

    if "Date" in event_df.columns and "date" not in event_df.columns:
        event_df = event_df.rename(columns={"Date": "date"})

    if "date" not in event_df.columns:
        if "start" in event_df.columns:
            event_df["date"] = pd.to_datetime(event_df["start"], errors="coerce").dt.normalize()
        else:
            raise ValueError(f"Could not find date/start column in {path}")
    else:
        event_df["date"] = pd.to_datetime(event_df["date"], errors="coerce").dt.normalize()

    if "estimated_attendance" not in event_df.columns:
        event_df["estimated_attendance"] = default_attendance
    else:
        event_df["estimated_attendance"] = pd.to_numeric(
            event_df["estimated_attendance"],
            errors="coerce",
        ).fillna(default_attendance)

    if hour_column and hour_column in event_df.columns:
        event_df["event_hour"] = pd.to_numeric(event_df[hour_column], errors="coerce")
    elif "start_hour" in event_df.columns:
        event_df["event_hour"] = pd.to_numeric(event_df["start_hour"], errors="coerce")
    elif "hour" in event_df.columns:
        event_df["event_hour"] = pd.to_numeric(event_df["hour"], errors="coerce")
    elif "start" in event_df.columns:
        event_df["event_hour"] = pd.to_datetime(event_df["start"], errors="coerce").dt.hour
    else:
        event_df["event_hour"] = 18

    event_df["event_hour"] = event_df["event_hour"].fillna(18).astype(int)
    event_df["venue_type"] = venue_type
    event_df = event_df.dropna(subset=["date"])

    return event_df[["date", "estimated_attendance", "event_hour", "venue_type"]]


scg = load_event_file(SCG_FILE, default_attendance=35000, venue_type=2, hour_column="start_hour")
hordern = load_event_file(HORDERN_FILE, default_attendance=5500, venue_type=0, hour_column="hour")
randwick = load_event_file(RANDWICK_FILE, default_attendance=20000, venue_type=1, hour_column="hour")
allianz = load_event_file(ALLIANZ_FILE, default_attendance=40000, venue_type=2, hour_column="start_hour")

events = pd.concat([scg, hordern, randwick, allianz], ignore_index=True)

print(f"\nTotal events across all venues: {len(events):,}")
print(events.groupby(events["date"].dt.year)["estimated_attendance"].count().rename("event_count"))


# ============================================================
# Build event hours with pre/post event windows
# ============================================================

event_hours = []

for _, row in events.iterrows():
    start = int(row["event_hour"])
    duration = 3
    end = start + duration
    pre_start = max(0, start - 2)      # 2 hours before event
    post_end = min(24, end + 3)        # 3 hours after event

    for h in range(pre_start, post_end):
        event_hours.append(
            {
                "date": row["date"],
                "hour": h,
                "has_event": 1,
                "max_attendance": row["estimated_attendance"],
                "total_attendance": row["estimated_attendance"],
                "venue_type": row["venue_type"],
            }
        )

if event_hours:
    event_hours_df = pd.DataFrame(event_hours).groupby(["date", "hour"]).agg(
        has_event=("has_event", "max"),
        max_attendance=("max_attendance", "max"),
        total_attendance=("total_attendance", "sum"),
        venue_type=("venue_type", "max"),
    ).reset_index()
else:
    event_hours_df = pd.DataFrame(
        columns=["date", "hour", "has_event", "max_attendance", "total_attendance", "venue_type"]
    )


# ============================================================
# Merge traffic and event data
# ============================================================

hourly = hourly.merge(event_hours_df, on=["date", "hour"], how="left")
hourly["has_event"] = hourly["has_event"].fillna(0)
hourly["max_attendance"] = hourly["max_attendance"].fillna(0)
hourly["total_attendance"] = hourly["total_attendance"].fillna(0)
hourly["venue_type"] = hourly["venue_type"].fillna(-1)


# ============================================================
# Feature engineering
# ============================================================

hourly["day_of_year"] = hourly["date"].dt.dayofyear
hourly["is_weekend"] = (hourly["day_of_week"] >= 6).astype(int)
hourly["public_holiday"] = hourly["public_holiday"].astype(int)
hourly["school_holiday"] = hourly["school_holiday"].astype(int)

FEATURES = [
    "hour",
    "month",
    "day_of_week",
    "day_of_year",
    "is_weekend",
    "public_holiday",
    "school_holiday",
    "has_event",
    "max_attendance",
    "total_attendance",
    "venue_type",
]
TARGET = "vehicle_count"

for col in FEATURES + [TARGET]:
    hourly[col] = pd.to_numeric(hourly[col], errors="coerce")

hourly = hourly.dropna(subset=FEATURES + [TARGET])


# ============================================================
# Train/test split and balancing
# ============================================================

train = hourly[(hourly["year"] >= 2013) & (hourly["year"] <= 2018)].copy()
test = hourly[hourly["year"] == 2019].copy()

if len(train) == 0 or len(test) == 0:
    raise ValueError("Train or test data is empty. Check traffic years 2013-2019.")

event_train = train[train["has_event"] == 1]
non_event_train = train[train["has_event"] == 0]

if len(event_train) > 0:
    sample_n = min(len(non_event_train), len(event_train) * 10)
    non_event_sample = non_event_train.sample(n=sample_n, random_state=42)
    train_balanced = pd.concat([event_train, non_event_sample]).sample(frac=1, random_state=42)
else:
    print("\nWarning: no event hours found in training data. Training on all rows.")
    train_balanced = train.copy()

X_train = train_balanced[FEATURES]
y_train = train_balanced[TARGET]
X_test = test[FEATURES]
y_test = test[TARGET]

print(f"\nEvent hours in training    : {len(event_train):,}")
print(f"Non-event hours in training: {len(non_event_train):,}")
print(f"Training rows used        : {len(X_train):,}")
print(f"Test rows                 : {len(X_test):,}")
print(f"Event hours in test       : {test['has_event'].sum():.0f}")


# ============================================================
# Train Random Forest
# ============================================================

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42,
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

test = test.copy()
test["predicted"] = y_pred

rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print(f"RMSE: {rmse:.0f} vehicles/hour")
print(f"MAE : {mae:.0f} vehicles/hour")
print(f"R²  : {r2:.3f}")

# Match main.py: subtract mean test-set bias from future predictions.
bias_correction = (y_pred - y_test).mean()
print(f"Bias correction: {bias_correction:.1f} vehicles/hour")


# ============================================================
# Prediction helpers copied/adapted from main.py
# ============================================================

def daily_congestion_level(total_vehicles):
    if total_vehicles < 10000:
        return "Low"
    if total_vehicles < 13000:
        return "Moderate"
    if total_vehicles < 15000:
        return "High"
    return "Severe"


def hourly_congestion_level(hourly_vehicles):
    if hourly_vehicles < 300:
        return "Low"
    if hourly_vehicles < 600:
        return "Moderate"
    if hourly_vehicles < 900:
        return "High"
    return "Severe"


def event_traffic_contribution(max_attendance, hour, event_hour, event_duration=3, venue_type=-1):
    """
    Estimate extra vehicles passing Station 55432 due to an event.
    Venue types:
      0 = Hordern
      1 = Randwick
      2 = SCG/Allianz
    """
    if max_attendance <= 0 or hour > 23:
        return 0

    drive_rate = max(0.13, 0.40 - (max_attendance / 40000) * 0.27)

    if venue_type == 0:
        capture_rate = 0.20 if max_attendance < 3000 else 0.30
    elif venue_type == 1:
        capture_rate = 0.60
    elif venue_type == 2:
        if max_attendance < 10000:
            capture_rate = 0.10
        elif max_attendance < 25000:
            capture_rate = 0.10 + (max_attendance - 10000) / 15000 * 0.20
        elif max_attendance < 30000:
            capture_rate = 0.30 + (max_attendance - 25000) / 5000 * 0.20
        else:
            capture_rate = min(0.50 + (max_attendance - 30000) / 10000 * 0.15, 0.65)
    else:
        capture_rate = 0.50

    cars = max_attendance * drive_rate * capture_rate
    post_start = event_hour + event_duration

    if hour == event_hour - 2:
        return cars * 0.15
    if hour == event_hour - 1:
        return cars * 0.35
    if hour == event_hour:
        return cars * 0.20
    if event_hour < hour < post_start:
        return cars * 0.06
    if hour == post_start and post_start <= 23:
        return cars * 0.15
    if hour == post_start + 1 and post_start + 1 <= 23:
        return cars * 0.10
    if hour == post_start + 2 and post_start + 2 <= 23:
        return cars * 0.05

    return 0


def predict_hour(date_str, hour, has_event=0, max_attendance=0, total_attendance=0,
                 venue_type=-1, event_hour=18, event_duration=3):
    date = pd.Timestamp(date_str)
    input_data = pd.DataFrame([
        {
            "hour": hour,
            "month": date.month,
            "day_of_week": date.dayofweek + 1,
            "day_of_year": date.dayofyear,
            "is_weekend": 1 if date.dayofweek >= 5 else 0,
            "public_holiday": 0,
            "school_holiday": 0,
            "has_event": has_event,
            "max_attendance": max_attendance,
            "total_attendance": total_attendance,
            "venue_type": venue_type,
        }
    ])

    base = max(0, model.predict(input_data)[0] - bias_correction)

    if has_event:
        extra = event_traffic_contribution(max_attendance, hour, event_hour, event_duration, venue_type)
        return base + extra

    return base


def predict_day(date_str, has_event=0, max_attendance=0, total_attendance=0,
                event_hour=18, event_duration=3, venue_type=-1):
    pre_start = event_hour - 2
    post_end = min(event_hour + event_duration + 3, 24)
    hourly_predictions = []
    base_predictions = []
    event_extras = []

    for h in range(24):
        h_event = has_event if (has_event and pre_start <= h < post_end) else 0
        vt = venue_type if h_event else -1

        base = predict_hour(date_str, h, 0, 0, 0, -1, event_hour, event_duration)
        total = predict_hour(
            date_str,
            h,
            h_event,
            max_attendance,
            total_attendance,
            vt,
            event_hour,
            event_duration,
        )

        base_predictions.append(base)
        hourly_predictions.append(total)
        event_extras.append(max(0, total - base))

    total_daily = float(np.sum(hourly_predictions))
    historic_average = get_historic_daily_average(date_str)
    difference = total_daily - historic_average
    percent_change = (difference / historic_average) * 100 if historic_average else 0
    peak_hour = int(np.argmax(hourly_predictions))
    peak_count = float(np.max(hourly_predictions))

    return {
        "date": pd.Timestamp(date_str).normalize(),
        "date_display": pd.Timestamp(date_str).strftime("%Y-%m-%d"),
        "historic_average": historic_average,
        "predicted_traffic": total_daily,
        "difference": difference,
        "percent_change": percent_change,
        "peak_hour": peak_hour,
        "peak_count": peak_count,
        "congestion": daily_congestion_level(total_daily),
        "hourly_predictions": hourly_predictions,
        "base_predictions": base_predictions,
        "event_extras": event_extras,
    }


# Historic daily average from training data, comparable to predicted daily totals.
train_daily = train.groupby("date").agg(
    total_vehicles=(TARGET, "sum"),
    month=("month", "first"),
    day_of_week=("day_of_week", "first"),
).reset_index()

daily_historic_lookup = train_daily.groupby(["month", "day_of_week"])["total_vehicles"].mean().reset_index()
daily_historic_lookup = daily_historic_lookup.rename(columns={"total_vehicles": "historic_average"})
overall_daily_average = train_daily["total_vehicles"].mean()
ASSISTANCE_THRESHOLD = train_daily["total_vehicles"].quantile(0.80)

print(f"\nAssistance threshold: {ASSISTANCE_THRESHOLD:,.0f} vehicles/day")


def get_historic_daily_average(date_str):
    date = pd.Timestamp(date_str)
    match = daily_historic_lookup[
        (daily_historic_lookup["month"] == date.month)
        & (daily_historic_lookup["day_of_week"] == date.dayofweek + 1)
    ]

    if len(match) == 0:
        return float(overall_daily_average)

    return float(match["historic_average"].iloc[0])


def infer_venue_type(venue_text):
    venue_text = str(venue_text).lower()
    if "scg" in venue_text or "allianz" in venue_text or "football stadium" in venue_text:
        return 2
    if "randwick" in venue_text:
        return 1
    if "hordern" in venue_text:
        return 0
    return 0


# ============================================================
# Load future event data and build display rows
# ============================================================

future_events_file = None
for pattern in ["events_cleaned_AEDT.csv", "events_cleaned.csv"]:
    try:
        future_events_file = find_file(pattern)
        break
    except FileNotFoundError:
        pass

future_scenarios = []

def add_future_scenario(date_str, event_name="No event", has_event=0, max_attendance=0,
                        event_hour=18, event_duration=3, venue_type=-1):
    result = predict_day(
        date_str,
        has_event=has_event,
        max_attendance=max_attendance,
        total_attendance=max_attendance,
        event_hour=event_hour,
        event_duration=event_duration,
        venue_type=venue_type,
    )
    result["event_name"] = event_name
    result["has_event"] = has_event
    result["max_attendance"] = max_attendance
    result["event_hour"] = event_hour
    result["event_duration"] = event_duration
    result["venue_type"] = venue_type
    result["assistance_needed"] = result["predicted_traffic"] > ASSISTANCE_THRESHOLD
    result["vehicles_above_threshold"] = result["predicted_traffic"] - ASSISTANCE_THRESHOLD
    future_scenarios.append(result)


if future_events_file is not None:
    print("Future events file:", future_events_file)
    future_events = pd.read_csv(future_events_file)
    future_events["date"] = pd.to_datetime(future_events["start"], errors="coerce").dt.normalize()
    future_events["event_hour"] = pd.to_datetime(future_events["start"], errors="coerce").dt.hour

    # Same headline dates used by main.py, with one manual comparison day.
    dates_to_predict = ["2026-05-30", "2026-06-01", "2027-09-13"]

    for date_str in dates_to_predict:
        date = pd.Timestamp(date_str).normalize()
        events_that_day = future_events[future_events["date"] == date]

        if len(events_that_day) > 0:
            first_event = events_that_day.iloc[0]
            event_name = str(first_event.get("summary", "Event"))
            event_hour = int(first_event.get("event_hour", 18)) if pd.notna(first_event.get("event_hour", 18)) else 18
            venue = first_event.get("venue", "")
            venue_type = infer_venue_type(venue)

            if "intensity" in events_that_day.columns:
                max_attendance = float(events_that_day["intensity"].max()) * 35000
            elif "estimated_attendance" in events_that_day.columns:
                max_attendance = float(pd.to_numeric(events_that_day["estimated_attendance"], errors="coerce").max())
            else:
                max_attendance = 35000

            add_future_scenario(
                date_str,
                event_name=event_name,
                has_event=1,
                max_attendance=max_attendance,
                event_hour=event_hour,
                event_duration=3,
                venue_type=venue_type,
            )
        else:
            add_future_scenario(date_str, event_name="No event", has_event=0)
else:
    print("\nWarning: could not find a future events file. Using manual scenarios instead.")
    add_future_scenario(
        "2026-05-30",
        event_name="Manual SCG/Allianz event",
        has_event=1,
        max_attendance=35000,
        event_hour=19,
        event_duration=3,
        venue_type=2,
    )
    add_future_scenario("2026-06-01", event_name="No event", has_event=0)
    add_future_scenario(
        "2027-09-13",
        event_name="Manual SCG/Allianz event",
        has_event=1,
        max_attendance=35000,
        event_hour=19,
        event_duration=3,
        venue_type=2,
    )


display_df = pd.DataFrame(future_scenarios)


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
            "peak_hour",
            "peak_count",
            "congestion",
            "assistance_needed",
        ]
    ].to_string(index=False)
)


# ============================================================
# Matplotlib assistance display
# ============================================================

fig, ax = plt.subplots(figsize=(14, 8))
ax.axis("off")

fig.suptitle(
    "Future Traffic Forecast Assistance Display",
    fontsize=18,
    fontweight="bold",
)

station_text = "Station 55432 — Cleveland Street, West of Anzac Parade"
ax.text(0.5, 0.94, station_text, ha="center", va="center", fontsize=12)

latest = display_df.iloc[0]
assistance_text = "YES" if latest["assistance_needed"] else "NO"
event_time_text = "No event"
if latest["has_event"]:
    event_time_text = (
        f"Event window: {int(latest['event_hour']) - 2:02d}:00 → "
        f"{min(int(latest['event_hour']) + int(latest['event_duration']) + 3, 23):02d}:00"
    )

summary_text = (
    f"Selected Date: {latest['date_display']}\n"
    f"Scenario: {latest['event_name']}\n"
    f"{event_time_text}\n\n"
    f"Historic Average: {latest['historic_average']:,.0f} vehicles/day\n"
    f"Predicted Traffic: {latest['predicted_traffic']:,.0f} vehicles/day\n"
    f"Difference: {latest['difference']:+,.0f} vehicles/day\n"
    f"Change: {latest['percent_change']:+.1f}%\n"
    f"Peak Hour: {int(latest['peak_hour']):02d}:00 ({latest['peak_count']:,.0f} vehicles/hour)\n"
    f"Congestion Level: {latest['congestion']}\n"
    f"Assistance Needed: {assistance_text}"
)

ax.text(
    0.5,
    0.68,
    summary_text,
    ha="center",
    va="center",
    fontsize=13,
    bbox=dict(boxstyle="round,pad=0.8", edgecolor="black", facecolor="white"),
)

table_data = []

for _, row in display_df.iterrows():
    table_data.append(
        [
            row["date_display"],
            str(row["event_name"])[:30],
            f"{row['historic_average']:,.0f}",
            f"{row['predicted_traffic']:,.0f}",
            f"{row['difference']:+,.0f}",
            f"{row['percent_change']:+.1f}%",
            f"{int(row['peak_hour']):02d}:00",
            row["congestion"],
            "YES" if row["assistance_needed"] else "NO",
        ]
    )

column_labels = [
    "Date",
    "Event",
    "Historic Avg",
    "Predicted",
    "Difference",
    "% Change",
    "Peak",
    "Congestion",
    "Assist?",
]

table = ax.table(
    cellText=table_data,
    colLabels=column_labels,
    loc="lower center",
    cellLoc="center",
    colLoc="center",
    bbox=[0.01, 0.05, 0.98, 0.36],
)

table.auto_set_font_size(False)
table.set_fontsize(8.5)
table.scale(1, 1.5)

plt.tight_layout()

output_file = BASE_DIR / "future_traffic_forecast_display.png"
plt.savefig(output_file, dpi=300)
print(f"\nSaved display to: {output_file}")

plt.show()
