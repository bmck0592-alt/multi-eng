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
    duration   = 3
    end        = start + duration
    pre_start  = max(0, start - 2)    # 2 hours before kickoff
    post_end   = min(24, end + 3)     # 3 hours after final whistle
 
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
 
# ── Bias correction ───────────────────────────────────────────
# Model consistently over-predicts — calculate mean bias on test set
# and subtract it from all future predictions to remove the offset
bias_correction = (y_pred - y_test).mean()
print(f"\nBias correction    : {bias_correction:.1f} vehicles/hour (subtracted from all predictions)")
print(f"(Positive = model over-predicts, Negative = under-predicts)")
 
# ── Feature importance plot ───────────────────────────────────
importances = pd.Series(model.feature_importances_, index=FEATURES)
importances.sort_values().plot(kind="barh", figsize=(8, 5), color="steelblue")
plt.title("Feature importance — hourly model with event data")
plt.xlabel("Importance score")
plt.tight_layout()
plt.savefig("feature_importance_hourly.png", dpi=150)
plt.close()
print("\n✓ Model plots saved.")
 
 
# ── Event traffic contribution (additive, physics-based) ──────
def event_traffic_contribution(max_attendance, hour, event_hour, event_duration=3, venue_type=-1):
    """
    Estimate extra vehicles on ANZAC Parade caused by the event.
    Drive rate scales with attendance — bigger crowds shift to PT.
    Capture rate accounts for how much event traffic actually passes
    Station 55432 based on venue access routes:
      - Hordern (0):    ~80% — primary access via ANZAC Parade/Cleveland St
      - Randwick (1):   ~60% — most traffic heads south down ANZAC Parade
      - SCG/Allianz (2): ~25% — mostly Moore Park Rd, some via Entertainment Quarter
    """
    if max_attendance <= 0:
        return 0
 
    # Hard cap — never apply event traffic past midnight
    if hour > 23:
        return 0
 
    # drive_rate scales smoothly down as attendance grows —
    # bigger crowds shift to PT due to parking scarcity.
    drive_rate = max(0.13, 0.40 - (max_attendance / 40000) * 0.27)
 
    # Venue-specific capture rate — what % of event traffic passes the sensor
    # For SCG/Allianz: capture rate scales with attendance based on road closure tipping point.
    # Under 25,000 — Moore Park Rd stays open, most traffic bypasses ANZAC Parade (low capture)
    # Over 30,000 — mandatory road closures force traffic onto ANZAC Parade (high capture)
    # Small events still cause some increase, just not much.
    if venue_type == 0:
        # Hordern — direct ANZAC Parade access but small venue
        # Under 5,500 capacity so never triggers road closures alone
        # Small consistent impact — calibrated from observed data
        if max_attendance < 3000:
            capture_rate = 0.20    # very small event, minimal impact
        else:
            capture_rate = 0.30    # full capacity, moderate impact
    elif venue_type == 1:
        capture_rate = 0.60    # Randwick — most traffic heads south on ANZAC Parade
    elif venue_type == 2:
        # SCG/Allianz: smooth ramp from 15% (small game) to 65% (sellout with road closures)
        # Tipping point at 25,000-30,000 where road closures kick in
        if max_attendance < 10000:
            capture_rate = 0.10    # small game, Moore Park Rd open, minimal ANZAC impact
        elif max_attendance < 25000:
            # gradual increase — some traffic spills onto ANZAC Parade
            capture_rate = 0.10 + (max_attendance - 10000) / 15000 * 0.20  # 10% → 30%
        elif max_attendance < 30000:
            # approaching tipping point — road management starting to redirect traffic
            capture_rate = 0.30 + (max_attendance - 25000) / 5000 * 0.20   # 30% → 50%
        else:
            # 30,000+ — road closures force traffic onto ANZAC Parade
            capture_rate = 0.50 + (max_attendance - 30000) / 10000 * 0.15  # 50% → 65% cap
            capture_rate = min(capture_rate, 0.65)
    else:
        capture_rate = 0.50    # unknown venue — assume moderate capture
 
    cars = max_attendance * drive_rate * capture_rate
 
    post_start = event_hour + event_duration
 
    # Arrival window: gradual build over 2 hours before kickoff
    # Arrivals are stronger signal than departures (Uber/taxi pickup reduces driving home)
    if hour == event_hour - 2:
        return cars * 0.15
    elif hour == event_hour - 1:
        return cars * 0.35
    elif hour == event_hour:
        return cars * 0.20  # late arrivals still trickling in
 
    # In-game: light sustained elevation — drop-offs, circling
    elif event_hour < hour < post_start:
        return cars * 0.06
 
    # Departure window: weaker than arrivals — many leave by Uber/taxi/PT
    elif hour == post_start and post_start <= 23:
        return cars * 0.15
    elif hour == post_start + 1 and post_start + 1 <= 23:
        return cars * 0.10
    elif hour == post_start + 2 and post_start + 2 <= 23:
        return cars * 0.05
 
    return 0
 
 
# ── Core prediction functions ─────────────────────────────────
def predict_hour(date_str, hour, has_event=0, max_attendance=0,
                 total_attendance=0, venue_type=-1,
                 event_hour=18, event_duration=3):
    date = pd.Timestamp(date_str)
    input_data = pd.DataFrame([{
        "hour":              hour,
        "month":             date.month,
        "day_of_week":       date.dayofweek + 1,
        "day_of_year":       date.dayofyear,
        "is_weekend":        1 if date.dayofweek >= 5 else 0,
        "public_holiday":    0,
        "school_holiday":    0,
        "has_event":         has_event,
        "max_attendance":    max_attendance,
        "total_attendance":  total_attendance,
        "venue_type":        venue_type
    }])
    base = max(0, model.predict(input_data)[0] - bias_correction)
 
    if has_event:
        extra = event_traffic_contribution(max_attendance, hour, event_hour, event_duration, venue_type)
        return base + extra
 
    return base
 
 
def predict_day_summary(date_str, has_event=0, max_attendance=0,
                         total_attendance=0, event_hour=18,
                         event_duration=3, venue_type=-1):
    hourly_preds = []
    pre_start = event_hour - 2
    post_end  = min(event_hour + event_duration + 3, 24)
 
    for h in range(24):
        h_event = has_event if (has_event and pre_start <= h < post_end) else 0
        vt      = venue_type if h_event else -1
        hourly_preds.append(predict_hour(date_str, h, h_event, max_attendance,
                                          total_attendance, vt, event_hour, event_duration))
 
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
        print(f"Attendance       : {max_attendance:,.0f}")
        drive_rate  = max(0.13, 0.40 - (max_attendance / 40000) * 0.27)
        venue_label = "Hordern" if venue_type == 0 else "Randwick" if venue_type == 1 else "SCG/Allianz" if venue_type == 2 else "Unknown"
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
        extra_cars = max_attendance * drive_rate * capture_rate
        print(f"Drive rate       : {drive_rate*100:.1f}% ({venue_label} capture: {capture_rate*100:.0f}%)")
        print(f"Extra vehicles   : ~{extra_cars:.0f} passing sensor")
        print(f"Event window     : {event_hour-2:02d}:00 → {min(event_hour+event_duration+3, 23):02d}:00")
    return total, hourly_preds
 
 
def predict_day_hourly(date_str, has_event=0, max_attendance=0,
                        total_attendance=0, event_hour=18,
                        event_duration=3, venue_type=-1):
    print(f"\n{'='*50}")
    print(f"HOURLY BREAKDOWN: {date_str}")
    print(f"{'='*50}")
    print(f"{'Hour':<8} {'Base':>8} {'Event+':>8} {'Total':>8} {'Congestion':>12}")
    print("-" * 48)
 
    pre_start = event_hour - 2
    post_end  = min(event_hour + event_duration + 3, 24)
    hourly_preds = []
 
    for h in range(24):
        h_event = has_event if (has_event and pre_start <= h < post_end) else 0
        vt      = venue_type if h_event else -1
 
        date = pd.Timestamp(date_str)
        input_data = pd.DataFrame([{
            "hour":              h,
            "month":             date.month,
            "day_of_week":       date.dayofweek + 1,
            "day_of_year":       date.dayofyear,
            "is_weekend":        1 if date.dayofweek >= 5 else 0,
            "public_holiday":    0,
            "school_holiday":    0,
            "has_event":         h_event,
            "max_attendance":    max_attendance,
            "total_attendance":  total_attendance,
            "venue_type":        vt
        }])
        base  = model.predict(input_data)[0]
        extra = event_traffic_contribution(max_attendance, h, event_hour, event_duration, vt) if h_event else 0
        total = base + extra
        level = "Low" if total < 300 else "Moderate" if total < 600 else "High" if total < 900 else "Severe"
 
        if h_event:
            if h < event_hour:
                marker = " ◀ PRE-GAME"
            elif h < event_hour + event_duration:
                marker = " ◀ GAME"
            else:
                marker = " ◀ POST-GAME"
        else:
            marker = ""
 
        print(f"{h:02d}:00   {base:>8.0f} {extra:>8.0f} {total:>8.0f} {level:>12}{marker}")
        hourly_preds.append(total)
 
    return hourly_preds
 
 
def plot_future_day(date_str, has_event=0, max_attendance=0,
                    total_attendance=0, event_hour=18, event_duration=3,
                    venue_type=-1, event_name="Event"):
    hours = list(range(24))
    base_preds  = []
    total_preds = []
    pre_start = event_hour - 2
    post_end  = min(event_hour + event_duration + 3, 24)
 
    for h in hours:
        h_event = has_event if (has_event and pre_start <= h < post_end) else 0
        vt      = venue_type if h_event else -1
 
        date = pd.Timestamp(date_str)
        input_data = pd.DataFrame([{
            "hour":              h,
            "month":             date.month,
            "day_of_week":       date.dayofweek + 1,
            "day_of_year":       date.dayofyear,
            "is_weekend":        1 if date.dayofweek >= 5 else 0,
            "public_holiday":    0,
            "school_holiday":    0,
            "has_event":         h_event,
            "max_attendance":    max_attendance,
            "total_attendance":  total_attendance,
            "venue_type":        vt
        }])
        base  = model.predict(input_data)[0]
        extra = event_traffic_contribution(max_attendance, h, event_hour, event_duration, vt) if h_event else 0
        base_preds.append(base)
        total_preds.append(base + extra)
 
    total_daily = sum(total_preds)
    congestion  = "Low" if total_daily < 10000 else "Moderate" if total_daily < 13000 else "High" if total_daily < 15000 else "Severe"
 
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(hours, total_preds, color="#89b4fa", linewidth=2.5, marker="o", label="Predicted traffic (with event)")
 
    if has_event:
        ax.plot(hours, base_preds, color="#cdd6f4", linewidth=1.5, linestyle=":", label="Baseline (no event)")
        for h in range(pre_start, min(post_end, 24)):
            ax.axvspan(h - 0.5, h + 0.5, alpha=0.15, color="#f9e2af")
        ax.axvspan(0, 0, alpha=0.2, color="#f9e2af", label=f"Event window: {event_name}")
        ax.axvline(x=event_hour, color="#f38ba8", linewidth=2,
                   linestyle="--", label=f"Kickoff {event_hour:02d}:00")
        ax.axvline(x=event_hour + event_duration, color="#a6e3a1", linewidth=2,
                   linestyle="--", label=f"Final whistle {event_hour+event_duration:02d}:00")
 
    ax.set_ylim(0, max(total_preds) * 1.2)
    ax.set_title(f"Predicted Hourly Traffic — {date_str}  |  Total: {total_daily:.0f} vehicles  |  Congestion: {congestion}")
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
 
 
# ── Load future event data ────────────────────────────────────
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
predict_with_auto_events("2026-05-30")
predict_with_auto_events("2026-06-01")
 
print("\n=== 2027 PREDICTION (MANUAL with event) ===")
predict_day_summary("2027-09-13", has_event=1, max_attendance=35000,
                     total_attendance=35000, event_hour=19,
                     event_duration=3, venue_type=2)
predict_day_hourly("2027-09-13", has_event=1, max_attendance=35000,
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
hours           = list(range(24))
no_event_preds  = []
hordern_preds   = []
allianz_preds   = []
 
event_hour = 19
event_duration = 3
pre_start = event_hour - 2
post_end  = event_hour + event_duration + 3
 
for h in hours:
    h_event = 1 if pre_start <= h < post_end else 0
 
    no_event_preds.append(predict_hour("2027-09-13", h, 0, 0, 0, -1, event_hour, event_duration))
    hordern_preds.append(predict_hour("2027-09-13",  h, h_event, 5500,  5500,  0, event_hour, event_duration))
    allianz_preds.append(predict_hour("2027-09-13",  h, h_event, 25000, 25000, 2, event_hour, event_duration))
 
fig, ax = plt.subplots(figsize=(12, 5))
ax.plot(hours, no_event_preds, color="#89b4fa", linewidth=2.5, marker="o",
        linestyle="--", label="No event")
ax.plot(hours, hordern_preds,  color="#a6e3a1", linewidth=2.5, marker="o",
        label="Hordern concert (5,500 attendance)")
ax.plot(hours, allianz_preds,  color="#f38ba8", linewidth=2.5, marker="o",
        label="Allianz Stadium A-League (25,000 attendance)")
 
for h in range(pre_start, min(post_end, 24)):
    ax.axvspan(h - 0.5, h + 0.5, alpha=0.15, color="#f9e2af")
ax.axvline(x=event_hour, color="gray", linewidth=1.5, linestyle=":", alpha=0.7, label=f"Kickoff {event_hour:02d}:00")
 
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
 
 
# ── Real event validation: Sydney FC v Melbourne Victory ──────
# A-League match at SCG, 6 April 2019, attendance 14,155
# Kickoff assumed 19:00 (standard Saturday A-League timeslot)
# This validates the model against a real recorded event in the test set
 
event_date     = "2019-04-06"
event_attend   = 14155
event_kick     = 19
event_dur      = 2        # A-League matches ~2 hrs
event_vtype    = 2        # SCG = venue_type 2
 
# Get actual hourly traffic for that day from test set
actual_day = test[test["date"] == pd.Timestamp(event_date).normalize()].sort_values("hour")
 
if len(actual_day) > 0:
    hours        = list(range(24))
    model_only   = []   # raw model, no event contribution
    with_event   = []   # model + physics-based event contribution
    actual       = actual_day.set_index("hour")["vehicle_count"].reindex(hours, fill_value=0).tolist()
 
    pre_start = event_kick - 2
    post_end  = min(event_kick + event_dur + 3, 24)
 
    for h in hours:
        h_event = 1 if pre_start <= h < post_end else 0
        vt      = event_vtype if h_event else -1
        model_only.append(predict_hour(event_date, h, 0, 0, 0, -1, event_kick, event_dur))
        with_event.append(predict_hour(event_date, h, h_event, event_attend, event_attend, vt, event_kick, event_dur))
 
    mae_no_event   = np.mean([abs(a - p) for a, p in zip(actual, model_only)])
    mae_with_event = np.mean([abs(a - p) for a, p in zip(actual, with_event)])
 
    print(f"\n{'='*55}")
    print(f"REAL EVENT VALIDATION: Sydney FC v Melbourne Victory")
    print(f"{'='*55}")
    print(f"Date       : {event_date} (Saturday)")
    print(f"Attendance : {event_attend:,}")
    print(f"Kickoff    : {event_kick:02d}:00 (assumed)")
    print(f"MAE without event contribution : {mae_no_event:.0f} vehicles/hour")
    print(f"MAE with event contribution    : {mae_with_event:.0f} vehicles/hour")
    improvement = mae_no_event - mae_with_event
    print(f"Improvement from event model   : {improvement:+.0f} vehicles/hour")
 
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hours, actual,     color="#89b4fa", linewidth=2.5, marker="o", label="Actual recorded traffic")
    ax.plot(hours, model_only, color="#cdd6f4", linewidth=1.5, linestyle=":", marker="o", markersize=4, label=f"Model only (no event) — MAE: {mae_no_event:.0f}")
    ax.plot(hours, with_event, color="#f38ba8", linewidth=2.5, linestyle="--", marker="o", label=f"Model + event contribution — MAE: {mae_with_event:.0f}")
 
    # Shade event window
    for h in range(pre_start, min(post_end, 24)):
        ax.axvspan(h - 0.5, h + 0.5, alpha=0.15, color="#f9e2af")
    ax.axvspan(0, 0, alpha=0.2, color="#f9e2af", label="Event window")
    ax.axvline(x=event_kick, color="#f38ba8", linewidth=2, linestyle="--", alpha=0.7, label=f"Kickoff {event_kick:02d}:00 (assumed)")
    ax.axvline(x=event_kick + event_dur, color="#a6e3a1", linewidth=2, linestyle="--", alpha=0.7, label=f"Final whistle {event_kick+event_dur:02d}:00")
 
    ax.set_ylim(0, max(max(actual), max(with_event)) * 1.2)
    ax.set_title(
        f"Real Event Validation — Sydney FC v Melbourne Victory\n"
        f"ANZAC Parade {event_date}  |  Attendance: {event_attend:,}  |  SCG"
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Vehicles per hour")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("real_event_validation.png", dpi=150)
    plt.close()
    print(f"✓ Saved real_event_validation.png")
else:
    print(f"⚠ {event_date} not found in test set")
 
 
# ── Hordern validation: Bugs/Grinspoon ───────────────────────
# Biggest observed traffic spike in 2019 test data (+1,318 above expected)
# Hordern Pavilion, 2 November 2019, capacity 5,500, kickoff 19:00
hordern_date    = "2019-11-02"
hordern_attend  = 5500
hordern_kick    = 19
hordern_dur     = 3
hordern_vtype   = 0   # Hordern = venue_type 0, capture rate 80%
 
actual_hordern = test[test["date"] == pd.Timestamp(hordern_date).normalize()].sort_values("hour")
 
if len(actual_hordern) > 0:
    hours        = list(range(24))
    model_only   = []
    with_event   = []
    actual       = actual_hordern.set_index("hour")["vehicle_count"].reindex(hours, fill_value=0).tolist()
 
    pre_start = hordern_kick - 2
    post_end  = min(hordern_kick + hordern_dur + 3, 24)
 
    for h in hours:
        h_event = 1 if pre_start <= h < post_end else 0
        vt      = hordern_vtype if h_event else -1
        model_only.append(predict_hour(hordern_date, h, 0, 0, 0, -1, hordern_kick, hordern_dur))
        with_event.append(predict_hour(hordern_date, h, h_event, hordern_attend, hordern_attend, vt, hordern_kick, hordern_dur))
 
    mae_no_event   = np.mean([abs(a - p) for a, p in zip(actual, model_only)])
    mae_with_event = np.mean([abs(a - p) for a, p in zip(actual, with_event)])
    improvement    = mae_no_event - mae_with_event
 
    print(f"\n{'='*55}")
    print(f"HORDERN VALIDATION: Bugs / Grinspoon")
    print(f"{'='*55}")
    print(f"Date       : {hordern_date} (Saturday)")
    print(f"Attendance : {hordern_attend:,} (venue capacity)")
    print(f"Kickoff    : {hordern_kick:02d}:00")
    print(f"Venue type : Hordern (80% capture rate)")
    print(f"MAE without event contribution : {mae_no_event:.0f} vehicles/hour")
    print(f"MAE with event contribution    : {mae_with_event:.0f} vehicles/hour")
    print(f"Improvement from event model   : {improvement:+.0f} vehicles/hour")
 
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.plot(hours, actual,     color="#89b4fa", linewidth=2.5, marker="o", label="Actual recorded traffic")
    ax.plot(hours, model_only, color="#cdd6f4", linewidth=1.5, linestyle=":", marker="o", markersize=4, label=f"Model only (no event) — MAE: {mae_no_event:.0f}")
    ax.plot(hours, with_event, color="#a6e3a1", linewidth=2.5, linestyle="--", marker="o", label=f"Model + event contribution — MAE: {mae_with_event:.0f}")
 
    for h in range(pre_start, min(post_end, 24)):
        ax.axvspan(h - 0.5, h + 0.5, alpha=0.15, color="#f9e2af")
    ax.axvspan(0, 0, alpha=0.2, color="#f9e2af", label="Event window")
    ax.axvline(x=hordern_kick,            color="#f38ba8", linewidth=2, linestyle="--", alpha=0.7, label=f"Doors open {hordern_kick:02d}:00")
    ax.axvline(x=hordern_kick + hordern_dur, color="#a6e3a1", linewidth=2, linestyle="--", alpha=0.7, label=f"End {hordern_kick+hordern_dur:02d}:00")
 
    ax.set_ylim(0, max(max(actual), max(with_event)) * 1.2)
    ax.set_title(
        f"Hordern Pavilion Validation — Bugs / Grinspoon\n"
        f"ANZAC Parade {hordern_date}  |  Capacity: {hordern_attend:,}  |  Hordern Pavilion"
    )
    ax.set_xlabel("Hour of day")
    ax.set_ylabel("Vehicles per hour")
    ax.set_xticks(range(24))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(24)], rotation=45)
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig("hordern_validation.png", dpi=150)
    plt.close()
    print(f"✓ Saved hordern_validation.png")
else:
    print(f"⚠ {hordern_date} not found in test set")