import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ── Load traffic data and reshape to hourly ───────────────────
traffic = pd.read_csv("tfnsw_hourly_traffic.csv")
traffic["date"] = pd.to_datetime(traffic["date"]).dt.tz_localize(None).dt.normalize()

hour_cols = [f"hour_{str(i).zfill(2)}" for i in range(24)]

hourly = traffic.melt(
    id_vars=["date", "year", "month", "day_of_week", "public_holiday", "school_holiday"],
    value_vars=hour_cols,
    var_name="hour_col",
    value_name="vehicle_count"
)
hourly["hour"] = hourly["hour_col"].str.extract(r"(\d+)").astype(int)
hourly = hourly.drop(columns="hour_col")
hourly = hourly.dropna(subset=["vehicle_count"])
hourly = hourly.sort_values(["date", "hour"]).reset_index(drop=True)

# ── Load and combine all event files ─────────────────────────
scg      = pd.read_csv("../../scg_events_2013_2019_clean.csv")
hordern  = pd.read_csv("../../hordern_events_2013_2019_clean.csv")
randwick = pd.read_csv("../../ben/royal_randwick_events_2013_2019_clean_with_estimated_attendance.csv")

scg = scg.rename(columns={"attendance": "estimated_attendance"})

scg["event_hour"]      = pd.to_numeric(scg["start_hour"], errors="coerce").fillna(18).astype(int)
hordern["event_hour"]  = pd.to_numeric(hordern["hour"],   errors="coerce").fillna(18).astype(int)
randwick["event_hour"] = pd.to_numeric(randwick["hour"],  errors="coerce").fillna(18).astype(int)

events = pd.concat([
    scg[["date", "estimated_attendance", "event_hour"]],
    hordern[["date", "estimated_attendance", "event_hour"]],
    randwick[["date", "estimated_attendance", "event_hour"]]
], ignore_index=True)
events["date"] = pd.to_datetime(events["date"]).dt.normalize()

event_hours = []
for _, row in events.iterrows():
    for h in range(int(row["event_hour"]), min(int(row["event_hour"]) + 5, 24)):
        event_hours.append({
            "date":             row["date"],
            "hour":             h,
            "has_event":        1,
            "max_attendance":   row["estimated_attendance"],
            "total_attendance": row["estimated_attendance"]
        })

event_hours_df = pd.DataFrame(event_hours).groupby(["date", "hour"]).agg(
    has_event        = ("has_event", "max"),
    max_attendance   = ("max_attendance", "max"),
    total_attendance = ("total_attendance", "sum")
).reset_index()

# ── Merge ─────────────────────────────────────────────────────
hourly = hourly.merge(event_hours_df, on=["date", "hour"], how="left")
hourly["has_event"]        = hourly["has_event"].fillna(0)
hourly["max_attendance"]   = hourly["max_attendance"].fillna(0)
hourly["total_attendance"] = hourly["total_attendance"].fillna(0)

# ── Feature engineering ───────────────────────────────────────
hourly["day_of_year"]    = hourly["date"].dt.dayofyear
hourly["is_weekend"]     = (hourly["day_of_week"] >= 6).astype(int)
hourly["public_holiday"] = hourly["public_holiday"].astype(int)
hourly["school_holiday"] = hourly["school_holiday"].astype(int)

FEATURES = [
    "hour", "month", "day_of_week", "day_of_year",
    "is_weekend", "public_holiday", "school_holiday",
    "has_event", "max_attendance", "total_attendance"
]
TARGET = "vehicle_count"

# ── Train/test split ──────────────────────────────────────────
train = hourly[(hourly["year"] >= 2017) & (hourly["year"] <= 2018)]
test  = hourly[hourly["year"] == 2019]

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"Training rows      : {len(X_train):,}")
print(f"Test rows          : {len(X_test):,}")
print(f"Event hours in test: {test['has_event'].sum():.0f}")

# ── Train model ───────────────────────────────────────────────
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nRMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.0f} vehicles/hour")
print(f"MAE  : {mean_absolute_error(y_test, y_pred):.0f}  vehicles/hour")
print(f"R²   : {r2_score(y_test, y_pred):.3f}")

test = test.copy()
test["predicted"] = y_pred
event_hours_test     = test[test["has_event"] == 1]
non_event_hours_test = test[test["has_event"] == 0]

if len(event_hours_test) > 0:
    print(f"\nEvent hours MAE    : {mean_absolute_error(event_hours_test['vehicle_count'], event_hours_test['predicted']):.0f}")
    print(f"Non-event hours MAE: {mean_absolute_error(non_event_hours_test['vehicle_count'], non_event_hours_test['predicted']):.0f}")

# ── Feature importance plot ───────────────────────────────────
importances = pd.Series(model.feature_importances_, index=FEATURES)
importances.sort_values().plot(kind="barh", figsize=(8, 5), color="steelblue")
plt.title("Feature importance — hourly model with event data")
plt.xlabel("Importance score")
plt.tight_layout()
plt.savefig("feature_importance_hourly.png", dpi=150)
plt.close()

print("\n✓ Plots saved.")

# ── Core prediction function (per hour) ───────────────────────
def predict_hour(date_str, hour, has_event=0, max_attendance=0, total_attendance=0):
    date = pd.Timestamp(date_str)
    input_data = pd.DataFrame([{
        "hour":             hour,
        "month":            date.month,
        "day_of_week":      date.dayofweek + 1,
        "day_of_year":      date.dayofyear,
        "is_weekend":       1 if date.dayofweek >= 5 else 0,
        "public_holiday":   0,
        "school_holiday":   0,
        "has_event":        has_event,
        "max_attendance":   max_attendance,
        "total_attendance": total_attendance
    }])
    return model.predict(input_data)[0]

# ── Daily summary prediction ──────────────────────────────────
def predict_day_summary(date_str, has_event=0, max_attendance=0,
                         total_attendance=0, event_hour=18):
    hourly_preds = []
    for h in range(24):
        h_event = has_event if (has_event and h >= event_hour) else 0
        hourly_preds.append(predict_hour(
            date_str, h, h_event, max_attendance, total_attendance
        ))

    total      = sum(hourly_preds)
    peak_hour  = hourly_preds.index(max(hourly_preds))
    peak_count = max(hourly_preds)
    congestion = "Low" if total < 10000 else "Moderate" if total < 13000 else "High" if total < 15000 else "Severe"

    print(f"\n{'='*50}")
    print(f"DATE SUMMARY: {date_str}")
    print(f"{'='*50}")
    print(f"Total vehicles   : {total:.0f}")
    print(f"Congestion level : {congestion}")
    print(f"Peak hour        : {peak_hour:02d}:00 ({peak_count:.0f} vehicles)")
    print(f"Event day        : {'Yes' if has_event else 'No'}")
    return total, hourly_preds

# ── Hourly breakdown prediction ───────────────────────────────
def predict_day_hourly(date_str, has_event=0, max_attendance=0,
                        total_attendance=0, event_hour=18):
    print(f"\n{'='*50}")
    print(f"HOURLY BREAKDOWN: {date_str}")
    print(f"{'='*50}")
    print(f"{'Hour':<8} {'Vehicles':>10} {'Congestion':>12}")
    print("-" * 32)

    hourly_preds = []
    for h in range(24):
        h_event = has_event if (has_event and h >= event_hour) else 0
        pred    = predict_hour(date_str, h, h_event, max_attendance, total_attendance)
        level   = "Low" if pred < 300 else "Moderate" if pred < 600 else "High" if pred < 900 else "Severe"
        marker  = " ◀ EVENT" if h_event else ""
        print(f"{h:02d}:00   {pred:>10.0f} {level:>12}{marker}")
        hourly_preds.append(pred)

    return hourly_preds

# ── Auto predict using Ben's future event data ────────────────
future_events = pd.read_csv("../../ben/events_cleaned.csv")
future_events["date"]       = pd.to_datetime(future_events["start"]).dt.tz_localize(None).dt.normalize()
future_events["event_hour"] = pd.to_datetime(future_events["start"]).dt.tz_localize(None).dt.hour

def predict_with_auto_events(date_str, show_hourly=True):
    date            = pd.Timestamp(date_str).normalize()
    events_that_day = future_events[future_events["date"] == date]

    if len(events_that_day) > 0:
        has_event        = 1
        max_attendance   = events_that_day["intensity"].max() * 50000
        total_attendance = max_attendance
        event_hour       = int(events_that_day["event_hour"].iloc[0])
        print(f"\nEvents on {date_str}:")
        for _, e in events_that_day.iterrows():
            print(f"  - {e['summary']} (starts {int(e['event_hour']):02d}:00)")
    else:
        has_event        = 0
        max_attendance   = 0
        total_attendance = 0
        event_hour       = 18
        print(f"\nNo events on {date_str}")

    predict_day_summary(date_str, has_event, max_attendance, total_attendance, event_hour)

    if show_hourly:
        predict_day_hourly(date_str, has_event, max_attendance, total_attendance, event_hour)

# ── Run predictions ───────────────────────────────────────────
print("\n=== AUTO EVENT PREDICTIONS ===")
predict_with_auto_events("2026-05-30", show_hourly=True)   # AFL game
predict_with_auto_events("2026-06-01", show_hourly=True)   # no event

print("\n=== MANUAL PREDICTIONS (2027) ===")
# Known event
predict_day_summary("2027-09-13", has_event=1, max_attendance=45000,
                     total_attendance=45000, event_hour=19)
predict_day_hourly("2027-09-13",  has_event=1, max_attendance=45000,
                    total_attendance=45000, event_hour=19)

# Normal day
predict_day_summary("2027-09-14")
predict_day_hourly("2027-09-14")
# ── Plot actual vs predicted for a specific past date ─────────
def plot_day(date_str):
    date = pd.Timestamp(date_str).normalize()
    sample = test[test["date"] == date].sort_values("hour")

    if len(sample) == 0:
        print(f"No data found for {date_str} — must be a date in 2019")
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sample["hour"], sample["vehicle_count"], label="Actual",
            color="#89b4fa", linewidth=2, marker="o")
    ax.plot(sample["hour"], sample["predicted"],     label="Predicted",
            color="#f38ba8", linewidth=2, linestyle="--", marker="o")

    # Shade event hours
    event_hrs = sample[sample["has_event"] == 1]["hour"].tolist()
    for h in event_hrs:
        ax.axvspan(h - 0.5, h + 0.5, alpha=0.15, color="yellow", label="_nolegend_")

    if event_hrs:
        ax.axvspan(0, 0, alpha=0.15, color="yellow", label="Event hours")

    ax.set_title(f"Actual vs Predicted by hour — {date_str}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Vehicles per hour")
    ax.set_xticks(range(24))
    ax.legend()
    plt.tight_layout()
    filename = f"hourly_plot_{date_str}.png"
    plt.savefig(filename, dpi=150)
    plt.show()
    print(f"✓ Saved {filename}")

# Try it — must be a date in 2019
plot_day("2019-03-02")
plot_day("2019-07-06")