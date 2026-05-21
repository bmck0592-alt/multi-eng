import pandas as pd

files = [
    'royal_randwick_events_2013_2019_clean_with_estimated_attendance.csv',
    'hordern_events_with_times.csv',
    'hordern_events.csv'
]

for f in files:
    try:
        df = pd.read_csv(f'../../ben/{f}')
        print(f'=== {f} ===')
        print(f'Columns: {df.columns.tolist()}')
        if 'year' in df.columns:
            print(f'Years: {sorted(df["year"].unique())}')
        print(df.head(3).to_string())
        print()
    except Exception as e:
        print(f'{f}: ERROR - {e}')
        # ── Auto predict using Ben's future event data ────────────────
future_events = pd.read_csv("../../ben/events_cleaned.csv")
future_events["date"] = pd.to_datetime(future_events["start"]).dt.normalize()

def predict_with_auto_events(date_str):
    date = pd.Timestamp(date_str).normalize()
    
    # Check if there's an event that day
    events_that_day = future_events[future_events["date"] == date]
    
    if len(events_that_day) > 0:
        has_event       = 1
        max_attendance  = events_that_day["intensity"].max() * 50000  # scale intensity to attendance
        total_attendance = max_attendance * len(events_that_day)
        print(f"Events found: {events_that_day['summary'].tolist()}")
    else:
        has_event        = 0
        max_attendance   = 0
        total_attendance = 0
        print("No events found that day")

    return predict_day(date_str, has_event, max_attendance, total_attendance)

# Try it
predict_with_auto_events("2026-05-30")  # AFL game in Ben's data
predict_with_auto_events("2026-06-01")  # random day