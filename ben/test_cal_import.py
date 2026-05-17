import requests
import pandas as pd
from icalendar import Calendar

url = "https://ics.ecal.com/ecal-sub/69e9a6f1389c6b0002f62d52/Venues%20NSW.ics"

response = requests.get(url)
cal = Calendar.from_ical(response.content)

events = []

for component in cal.walk():
    if component.name == "VEVENT":
        events.append({
            "summary": str(component.get("summary")),
            "start": component.get("dtstart").dt,
            "end": component.get("dtend").dt,
            "location": str(component.get("location"))
        })

df = pd.DataFrame(events)

df["start"] = pd.to_datetime(df["start"])
df["end"] = pd.to_datetime(df["end"])

df["hour"] = df["start"].dt.hour
df["day_of_week"] = df["start"].dt.dayofweek
df["month"] = df["start"].dt.month
df["year"] = df["start"].dt.year
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

df["duration_hours"] = (df["end"] - df["start"]).dt.total_seconds() / 3600

def map_location(loc):
    if pd.isna(loc):
        return "unknown"

    loc = str(loc).lower()

    if "scg" in loc or "cricket" in loc:
        return "scg"
    elif "randwick" in loc:
        return "randwick"
    elif "hordern" in loc:
        return "hordern"
    else:
        return "other"

df["venue"] = df["location"].apply(map_location)

def estimate_intensity(name):
    name = str(name).lower()

    if any(x in name for x in ["concert", "festival"]):
        return 3
    elif any(x in name for x in ["game", "match", "race", "racing"]):
        return 2
    else:
        return 1

df["intensity"] = df["summary"].apply(estimate_intensity)

expanded = []

for _, row in df.iterrows():
    times = pd.date_range(start=row["start"], end=row["end"], freq="h")

    for t in times:
        expanded.append({
            "time": t,
            "summary": row["summary"],
            "venue": row["venue"],
            "location": row["location"],
            "intensity": row["intensity"],
            "duration_hours": row["duration_hours"],
            "is_weekend": row["is_weekend"],
            "hour": t.hour,
            "day_of_week": t.dayofweek,
            "month": t.month,
            "year": t.year
        })

event_df = pd.DataFrame(expanded)

print(df.head())
print(event_df.head())

df.to_csv("events_cleaned.csv", index=False)
event_df.to_csv("events_hourly.csv", index=False)

print("Saved events_cleaned.csv and events_hourly.csv")