import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# ── Load traffic data ─────────────────────────────────────────
traffic = pd.read_csv("tfnsw_hourly_traffic.csv")
traffic["date"] = pd.to_datetime(traffic["date"]).dt.tz_localize(None).dt.normalize()

# ── Load and combine all event files ─────────────────────────
scg      = pd.read_csv("../../scg_events_2013_2019_clean.csv")
hordern  = pd.read_csv("../../hordern_events_2013_2019_clean.csv")
randwick = pd.read_csv("../../ben/royal_randwick_events_2013_2019_clean_with_estimated_attendance.csv")

scg = scg.rename(columns={"attendance": "estimated_attendance"})

events = pd.concat([
    scg[["date", "estimated_attendance"]],
    hordern[["date", "estimated_attendance"]],
    randwick[["date", "estimated_attendance"]]
], ignore_index=True)
events["date"] = pd.to_datetime(events["date"]).dt.normalize()

events_daily = events.groupby("date").agg(
    event_flag       = ("date", "count"),
    max_attendance   = ("estimated_attendance", "max"),
    total_attendance = ("estimated_attendance", "sum")
).reset_index()
events_daily["max_attendance"]   = events_daily["max_attendance"].fillna(0)
events_daily["total_attendance"] = events_daily["total_attendance"].fillna(0)

# ── Merge ─────────────────────────────────────────────────────
df = traffic.merge(events_daily, on="date", how="left")
df["event_flag"]       = df["event_flag"].fillna(0)
df["max_attendance"]   = df["max_attendance"].fillna(0)
df["total_attendance"] = df["total_attendance"].fillna(0)

# ── Feature engineering ───────────────────────────────────────
df["day_of_year"]    = df["date"].dt.dayofyear
df["is_weekend"]     = (df["day_of_week"] >= 6).astype(int)
df["public_holiday"] = df["public_holiday"].astype(int)
df["school_holiday"] = df["school_holiday"].astype(int)
df["has_event"]      = (df["event_flag"] > 0).astype(int)

FEATURES = [
    "month", "day_of_week", "day_of_year",
    "is_weekend", "public_holiday", "school_holiday",
    "has_event", "max_attendance", "total_attendance"
]
TARGET = "daily_total"

# ── Train/test split ──────────────────────────────────────────
train = df[(df["year"] >= 2017) & (df["year"] <= 2018)]
test  = df[df["year"] == 2019]

X_train, y_train = train[FEATURES], train[TARGET]
X_test,  y_test  = test[FEATURES],  test[TARGET]

print(f"Training rows     : {len(X_train):,}")
print(f"Test rows         : {len(X_test):,}")
print(f"Event days in test: {test['has_event'].sum()}")

# ── Train model ───────────────────────────────────────────────
model = RandomForestRegressor(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=5,
    random_state=42
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print(f"\nRMSE : {np.sqrt(mean_squared_error(y_test, y_pred)):.0f} vehicles/day")
print(f"MAE  : {mean_absolute_error(y_test, y_pred):.0f}  vehicles/day")
print(f"R²   : {r2_score(y_test, y_pred):.3f}")

test = test.copy()
test["predicted"] = y_pred
event_days     = test[test["has_event"] == 1]
non_event_days = test[test["has_event"] == 0]

if len(event_days) > 0:
    print(f"\nEvent days MAE    : {mean_absolute_error(event_days['daily_total'], event_days['predicted']):.0f}")
    print(f"Non-event days MAE: {mean_absolute_error(non_event_days['daily_total'], non_event_days['predicted']):.0f}")

# ── Feature importance plot ───────────────────────────────────
importances = pd.Series(model.feature_importances_, index=FEATURES)
importances.sort_values().plot(kind="barh", figsize=(8, 5), color="steelblue")
plt.title("Feature importance — model with event data")
plt.xlabel("Importance score")
plt.tight_layout()
plt.savefig("feature_importance_with_events.png", dpi=150)
plt.show()

# ── Actual vs predicted plot ──────────────────────────────────
test = test.sort_values("date")
fig, ax = plt.subplots(figsize=(14, 5))
ax.plot(test["date"], test["daily_total"], label="Actual",    color="#89b4fa", linewidth=1.5)
ax.plot(test["date"], test["predicted"],  label="Predicted", color="#f38ba8", linewidth=1.5, linestyle="--")
ax.set_title("Actual vs Predicted — ANZAC Parade 2019 (with event data)")
ax.set_xlabel("Date")
ax.set_ylabel("Vehicles per day")
ax.legend()
plt.tight_layout()
plt.savefig("actual_vs_predicted_with_events.png", dpi=150)
plt.show()

print("\n✓ Done.")
# ── Predict a specific day ────────────────────────────────────
def predict_day(date_str, has_event=0, max_attendance=0, total_attendance=0):
    date = pd.Timestamp(date_str)
    input_data = pd.DataFrame([{
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
    prediction = model.predict(input_data)[0]
    congestion = "Low" if prediction < 10000 else "Moderate" if prediction < 13000 else "High" if prediction < 15000 else "Severe"
    print(f"\nDate       : {date_str}")
    print(f"Predicted  : {prediction:.0f} vehicles")
    print(f"Congestion : {congestion}")
    return prediction
# Weekend SCG game
predict_day("2027-07-10", has_event=1, max_attendance=45000, total_attendance=45000)

# Same weekend day no event
predict_day("2027-07-10")
# Randwick race day
predict_day("2027-03-20", has_event=1, max_attendance=25000, total_attendance=25000)