import requests
import pandas as pd
import sqlite3
from icalendar import Calendar

# Allianz / Venues NSW calendar link
calendar_url = "webcal://ics.ecal.com/ecal-sub/69fc1a9f069a150002f4887d/Venues%20NSW.ics"

# Convert webcal to https so Python can read it
calendar_url = calendar_url.replace("webcal://", "https://")

response = requests.get(calendar_url)
response.raise_for_status()

cal = Calendar.from_ical(response.content)

events = []

for component in cal.walk():
    if component.name == "VEVENT":
        summary = component.get("summary")
        start = component.get("dtstart")
        end = component.get("dtend")
        location = component.get("location")
        description = component.get("description")

        events.append({
            "event_name": str(summary) if summary else None,
            "start": start.dt if start else None,
            "end": end.dt if end else None,
            "location": str(location) if location else None,
            "description": str(description) if description else None,
            "venue": "Allianz Stadium / Venues NSW",
            "source": calendar_url
        })

df = pd.DataFrame(events)

if df.empty:
    print("No events found in the calendar.")
else:
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")

    df["date"] = df["start"].dt.date
    df["time"] = df["start"].dt.strftime("%I:%M %p")
    df["hour"] = df["start"].dt.hour
    df["day_of_week"] = df["start"].dt.day_name()
    df["month"] = df["start"].dt.month
    df["year"] = df["start"].dt.year
    df["is_weekend"] = df["day_of_week"].isin(["Saturday", "Sunday"]).astype(int)

    df["duration_hours"] = (
        df["end"] - df["start"]
    ).dt.total_seconds() / 3600

    df = df[[
        "start",
        "end",
        "date",
        "time",
        "event_name",
        "venue",
        "location",
        "description",
        "hour",
        "day_of_week",
        "month",
        "year",
        "is_weekend",
        "duration_hours",
        "source"
    ]]

    print(df.head(30))
    print(f"\nFound {len(df)} events.")

    df.to_csv("allianz_venues_nsw_events.csv", index=False)

    conn = sqlite3.connect("events_database.db")

    df.to_sql(
        "allianz_venues_nsw_events",
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()

    print("\nSaved:")
    print("- allianz_venues_nsw_events.csv")
    print("- events_database.db")