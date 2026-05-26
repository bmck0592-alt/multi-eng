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

# ── Load all event files ──────────────────────────────────────
scg      = pd.read_csv("../../scg_events_2013_2019_clean.csv")
hordern  = pd.read_csv("../../hordern_events_2013_2019_clean.csv")
randwick = pd.read_csv("../../ben/royal_randwick_events_2013_2019_clean_with_estimated_attendance.csv")
allianz  = pd.read_csv("../../allianz_sydney_football_stadium_events_clean.csv")

# Rename and fill attendance with venue capacity estimates
scg     = scg.rename(columns={"attendance": "estimated_attendance"})
allianz = allianz.rename(columns={"attendance": "estimated_attendance"})

scg["estimated_attendance"]      = scg["estimated_attendance"].fillna(35000)
hordern["estimated_attendance"]  = hordern["estimated_attendance"].fillna(5500)
randwick["estimated_attendance"] = randwick["estimated_attendance"].fillna(20000)
allianz["estimated_attendance"]  = allianz["estimated_attendance"].fillna(40000)

# Get event hours and duration
scg["event_hour"]      = pd.to_numeric(scg["start_hour"],     errors="coerce").fillna(18).astype(int)
hordern["event_hour"]  = pd.to_numeric(hordern["hour"],       errors="coerce").fillna(18).astype(int)
randwick["event_hour"] = pd.to_numeric(randwick["hour"],      errors="coerce").fillna(18).astype(int)
allianz["event_hour"]  = pd.to_numeric(allianz["start_hour"], errors="coerce").fillna(18).astype(int)

# Add venue type
scg["venue_type"]      = 2
hordern["venue_type"]  = 0
randwick["venue_type"] = 1
allianz["venue_type"]  = 2

# ── Combine all events ────────────────────────────────────────
events = pd.concat([
    scg[["date", "estimated_attendance", "event_hour", "venue_type"]],
    hordern[["date", "estimated_attendance", "event_hour", "venue_type"]],
    randwick[["date", "estimated_attendance", "event_hour", "venue_type"]],
    allianz[["date", "estimated_attendance", "event_hour", "venue_type"]]
], ignore_index=True)
events["date"] = pd.to_datetime(events["date"]).dt.normalize()

print(f"Total events across all venues: {len(events)}")
print(events.groupby(pd.to_datetime(events["date"]).dt.year)["estimated_attendance"].count().rename("event_count"))

# ── Build event hours with pre/post game windows ──────────────
event_hours = []
for _, row in events.iterrows():
    start      = int(row["event_hour"])
    duration   = 3          # default game duration hours
    end        = start + duration
    pre_start  = max(0, start - 1)    # 1 hour before kickoff
    post_end   = min(24, end + 2)     # 2 hours after final whistle

    for h in range(pre_start, post_end):
        event_hours.append({
            "date":             row["date"],
            "hour":             h,
            "has_event":        1,
            "max_attendance":   row["estimated_attendance"],
            "total_attendance": row["estimated_attendance"],
            "venue_type":       row["venue_type"]
        })

event_hours_df = pd.DataFrame(event_hours).groupby(["date", "hour"]).agg(
    has_event        = ("has_event", "max"),
    max_attendance   = ("max_attendance", "max"),
    total_attendance = ("total_attendance", "sum"),
    venue_type       = ("venue_type", "max")
).reset_index()

# ── Merge ─────────────────────────────────────────────────────
hourly = hourly.merge(event_hours_df, on=["date", "hour"], how="left")
hourly["has_event"]        = hourly["has_event"].fillna(0)
hourly["max_attendance"]   = hourly["max_attendance"].fillna(0)
hourly["total_attendance"] = hourly["total_attendance"].fillna(0)
hourly["venue_type"]       = hourly["venue_type"].fillna(-1)

# ── Feature engineering ───────────────────────────────────────
hourly["day_of_year"]    = hourly["date"].dt.dayofyear
hourly["is_weekend"]     = (hourly["day_of_week"] >= 6).astype(int)
hourly["public_holiday"] = hourly["public_holiday"].astype(int)
hourly["school_holiday"] = hourly["school_holiday"].astype(int)

FEATURES = [
    "hour", "month", "day_of_week", "day_of_year",
    "is_weekend", "public_holiday", "school_holiday",
    "has_event", "max_attendance", "total_attendance", "venue_type"
]
TARGET = "vehicle_count"

# ── Train/test split ──────────────────────────────────────────
train = hourly[(hourly["year"] >= 2013) & (hourly["year"] <= 2018)]
test  = hourly[hourly["year"] == 2019]

# Undersample non-event hours
event_train     = train[train["has_event"] == 1]
non_event_train = train[train["has_event"] == 0].sample(
    n=len(event_train) * 10,
    random_state=42
)
train_balanced = pd.concat([event_train, non_event_train]).sample(frac=1, random_state=42)

X_train, y_train = train_balanced[FEATURES], train_balanced[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"\nEvent hours in training    : {len(event_train):,}")
print(f"Non-event hours in training: {len(non_event_train):,}")
print(f"\nTraining rows (balanced) : {len(X_train):,}")
print(f"Test rows                : {len(X_test):,}")
print(f"Event hours in test      : {test['has_event'].sum():.0f}")

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

print("\n✓ Model plots saved.")

# ── Attendance multiplier ─────────────────────────────────────
def apply_event_multiplier(base_pred, max_attendance):
    if max_attendance <= 0:
        return base_pred
    excess     = max(0, max_attendance - 5500)
    multiplier = 1 + (excess / 10000) * 0.05
    return base_pred * multiplier

# ── Core prediction functions ─────────────────────────────────
def predict_hour(date_str, hour, has_event=0, max_attendance=0,
                 total_attendance=0, venue_type=-1):
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
        "total_attendance": total_attendance,
        "venue_type":       venue_type
    }])
    base = model.predict(input_data)[0]
    if has_event:
        return apply_event_multiplier(base, max_attendance)
    return base

def predict_day_summary(date_str, has_event=0, max_attendance=0,
                         total_attendance=0, event_hour=18,
                         event_duration=3, venue_type=-1):
    hourly_preds = []
    for h in range(24):
        pre_start = event_hour - 1
        post_end  = event_hour + event_duration + 2
        h_event   = has_event if (has_event and pre_start <= h < post_end) else 0
        vt        = venue_type if h_event else -1
        hourly_preds.append(predict_hour(date_str, h, h_event, max_attendance, total_attendance, vt))
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
    if has_event:
        print(f"Event window     : {event_hour-1:02d}:00 → {event_hour+event_duration+2:02d}:00")
    return total, hourly_preds

def predict_day_hourly(date_str, has_event=0, max_attendance=0,
                        total_attendance=0, event_hour=18,
                        event_duration=3, venue_type=-1):
    print(f"\n{'='*50}")
    print(f"HOURLY BREAKDOWN: {date_str}")
    print(f"{'='*50}")
    print(f"{'Hour':<8} {'Vehicles':>10} {'Congestion':>12}")
    print("-" * 32)
    hourly_preds = []
    pre_start = event_hour - 1
    post_end  = event_hour + event_duration + 2
    for h in range(24):
        h_event = has_event if (has_event and pre_start <= h < post_end) else 0
        vt      = venue_type if h_event else -1
        pred    = predict_hour(date_str, h, h_event, max_attendance, total_attendance, vt)
        level   = "Low" if pred < 300 else "Moderate" if pred < 600 else "High" if pred < 900 else "Severe"
        if h_event:
            if h == pre_start:
                marker = " ◀ PRE-GAME"
            elif h >= event_hour + event_duration:
                marker = " ◀ POST-GAME"
            else:
                marker = " ◀ GAME"
        else:
            marker = ""
        print(f"{h:02d}:00   {pred:>10.0f} {level:>12}{marker}")
        hourly_preds.append(pred)
    return hourly_preds

def plot_future_day(date_str, has_event=0, max_attendance=0,
                    total_attendance=0, event_hour=18, event_duration=3,
                    venue_type=-1, event_name="Event"):
    hours = list(range(24))
    predictions = []
    pre_start = event_hour - 1
    post_end  = event_hour + event_duration + 2
    for h in hours:
        h_event = has_event if (has_event and pre_start <= h < post_end) else 0
        vt      = venue_type if h_event else -1
        predictions.append(predict_hour(date_str, h, h_event, max_attendance, total_attendance, vt))

    total      = sum(predictions)
    congestion = "Low" if total < 10000 else "Moderate" if total < 13000 else "High" if total < 15000 else "Severe"

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hours, predictions, color="#89b4fa", linewidth=2.5, marker="o", label="Predicted traffic")

    if has_event:
        for h in range(pre_start, min(post_end, 24)):
            ax.axvspan(h - 0.5, h + 0.5, alpha=0.2, color="#f9e2af")
        ax.axvspan(0, 0, alpha=0.2, color="#f9e2af", label=f"Event window: {event_name}")
        ax.axvline(x=event_hour, color="#f38ba8", linewidth=2,
                   linestyle="--", label=f"Kickoff {event_hour:02d}:00")
        ax.axvline(x=event_hour + event_duration, color="#a6e3a1", linewidth=2,
                   linestyle="--", label=f"Final whistle {event_hour+event_duration:02d}:00")

    ax.set_ylim(0, max(predictions) * 1.2)
    ax.set_title(f"Predicted Hourly Traffic — {date_str}  |  Total: {total:.0f} vehicles  |  Congestion: {congestion}")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Vehicles per hour")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
    ax.legend()
    plt.tight_layout()
    filename = f"predicted_hourly_{date_str}.png"
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✓ Saved {filename}")

# ── Load Ben's future event data ─────────────────────────────
future_events = pd.read_csv("../../ben/events_cleaned_AEDT.csv")
future_events["date"]       = pd.to_datetime(future_events["start"]).dt.normalize()
future_events["event_hour"] = pd.to_datetime(future_events["start"]).dt.hour

def predict_with_auto_events(date_str):
    date            = pd.Timestamp(date_str).normalize()
    events_that_day = future_events[future_events["date"] == date]

    if len(events_that_day) > 0:
        has_event        = 1
        max_attendance   = events_that_day["intensity"].max() * 35000
        total_attendance = max_attendance
        event_hour       = int(events_that_day["event_hour"].iloc[0])
        event_name       = events_that_day["summary"].iloc[0]
        venue            = events_that_day["venue"].iloc[0] if "venue" in events_that_day.columns else "scg"
        venue_type       = 2 if "scg" in str(venue).lower() or "allianz" in str(venue).lower() else 1 if "randwick" in str(venue).lower() else 0
        print(f"\nEvents on {date_str}:")
        for _, e in events_that_day.iterrows():
            print(f"  - {e['summary']} (starts {int(e['event_hour']):02d}:00)")
    else:
        has_event        = 0
        max_attendance   = 0
        total_attendance = 0
        event_hour       = 18
        event_name       = ""
        venue_type       = -1
        print(f"\nNo events on {date_str}")

    predict_day_summary(date_str, has_event, max_attendance, total_attendance,
                        event_hour, venue_type=venue_type)
    predict_day_hourly(date_str, has_event, max_attendance, total_attendance,
                       event_hour, venue_type=venue_type)
    plot_future_day(date_str, has_event, max_attendance, total_attendance,
                    event_hour, venue_type=venue_type, event_name=event_name)

# ── Run predictions ───────────────────────────────────────────
print("\n=== 2026 PREDICTIONS (AUTO) ===")
predict_with_auto_events("2026-05-30")   # has event
predict_with_auto_events("2026-06-01")   # no event

print("\n=== 2027 PREDICTION (MANUAL with event) ===")
predict_day_summary("2027-09-13", has_event=1, max_attendance=35000,
                     total_attendance=35000, event_hour=19,
                     event_duration=3, venue_type=2)
predict_day_hourly("2027-09-13",  has_event=1, max_attendance=35000,
                    total_attendance=35000, event_hour=19,
                    event_duration=3, venue_type=2)
plot_future_day("2027-09-13", has_event=1, max_attendance=35000,
                total_attendance=35000, event_hour=19, event_duration=3,
                venue_type=2, event_name="Major SCG Event")

print("\n=== 2027 PREDICTION (MANUAL no event) ===")
predict_day_summary("2027-09-13", has_event=0)
predict_day_hourly("2027-09-13",  has_event=0)
# ── Full year 2019 actual vs predicted plot ───────────────────
daily_test = test.copy()
daily_test = daily_test.groupby("date").agg(
    actual    = ("vehicle_count", "sum"),
    predicted = ("predicted", "sum")
).reset_index()

fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(daily_test["date"], daily_test["actual"],    label="Actual",    color="#89b4fa", linewidth=1.5)
ax.plot(daily_test["date"], daily_test["predicted"], label="Predicted", color="#f38ba8", linewidth=1.5, linestyle="--")
ax.set_title("Actual vs Predicted Daily Traffic — ANZAC Parade 2019")
ax.set_xlabel("Date")
ax.set_ylabel("Vehicles per day")
ax.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted_2019.png", dpi=150)
plt.close()
print("✓ Saved actual_vs_predicted_2019.png")
# ── Three way comparison: no event vs Hordern vs Allianz ──────
hours = list(range(24))
no_event_preds  = []
hordern_preds   = []
allianz_preds   = []

for h in hours:
    pre_start = 19
    post_end  = 19 + 3 + 2
    h_event = 1 if pre_start <= h < post_end else 0

    no_event_preds.append(predict_hour("2027-09-13", h, 0, 0, 0, -1))
    hordern_preds.append(predict_hour("2027-09-13", h, h_event, 5500, 5500, 0))
    allianz_preds.append(predict_hour("2027-09-13", h, h_event, 25000, 25000, 2))

fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(hours, no_event_preds, color="#89b4fa", linewidth=2.5, marker="o",
        linestyle="--", label="No event")
ax.plot(hours, hordern_preds,  color="#a6e3a1", linewidth=2.5, marker="o",
        label="Hordern concert (5,500 attendance)")
ax.plot(hours, allianz_preds,  color="#f38ba8", linewidth=2.5, marker="o",
        label="Allianz Stadium A-League (25,000 attendance)")

for h in range(18, 24):
    ax.axvspan(h - 0.5, h + 0.5, alpha=0.15, color="#f9e2af")
ax.axvline(x=19, color="gray", linewidth=1.5, linestyle=":", alpha=0.7, label="Kickoff 19:00")

ax.set_ylim(0, max(allianz_preds) * 1.2)
ax.set_title("Traffic Impact by Venue Size — ANZAC Parade 2027-09-13")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Vehicles per hour")
ax.set_xticks(range(24))
ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
ax.legend()
plt.tight_layout()
plt.savefig("venue_comparison.png", dpi=150)
plt.close()
print("✓ Saved venue_comparison.png")


# ── Actual vs predicted for a specific past day ───────────────
sample_date = pd.Timestamp("2019-10-19").normalize()
sample = test[test["date"] == sample_date].sort_values("hour")

if len(sample) > 0:
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(sample["hour"], sample["vehicle_count"], color="#89b4fa", linewidth=2.5,
            marker="o", label="Actual traffic")
    ax.plot(sample["hour"], sample["predicted"],     color="#f38ba8", linewidth=2.5,
            marker="o", linestyle="--", label="Predicted traffic")
    ax.set_ylim(0, max(sample["vehicle_count"].max(), sample["predicted"].max()) * 1.2)
    ax.set_title(f"Actual vs Predicted — ANZAC Parade {sample_date.date()}  |  MAE: {abs(sample['vehicle_count'] - sample['predicted']).mean():.0f} vehicles/hour")
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Vehicles per hour")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
    ax.legend()
    plt.tight_layout()
    plt.savefig("actual_vs_predicted_sample_day.png", dpi=150)
    plt.close()
    print(f"✓ Saved actual_vs_predicted_sample_day.png")
else:
    print("Date not in test set — try another 2019 date")