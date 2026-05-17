import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re

url = "https://sydneymusic.net/gig-guide/venues/hordern-pavilion"

headers = {
    "User-Agent": "Mozilla/5.0"
}

response = requests.get(url, headers=headers)
response.raise_for_status()

soup = BeautifulSoup(response.text, "html.parser")

text = soup.get_text("\n", strip=True)

with open("hordern_page_text.txt", "w", encoding="utf-8") as f:
    f.write(text)

lines = [line.strip() for line in text.split("\n") if line.strip()]

events = []

month_pattern = re.compile(
    r"^(January|February|March|April|May|June|July|August|September|October|November|December)\s+(\d{4})$",
    re.IGNORECASE
)

day_name_pattern = re.compile(
    r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)$",
    re.IGNORECASE
)

day_number_pattern = re.compile(
    r"^\d{1,2}$"
)

time_pattern = re.compile(
    r"^(\d{1,2}:\d{2}\s?(?:am|pm|AM|PM))"
)

current_month = None
current_year = None
current_day_name = None
current_day_number = None
current_event_name = None
current_support_act = None
in_upcoming_section = False

for line in lines:

    if line.startswith("Upcoming Gigs"):
        in_upcoming_section = True
        continue

    if line.startswith("Past Gigs"):
        in_upcoming_section = False
        break

    if not in_upcoming_section:
        continue

    month_match = month_pattern.match(line)
    if month_match:
        current_month = month_match.group(1)
        current_year = month_match.group(2)
        continue

    if day_name_pattern.match(line):
        current_day_name = line
        current_event_name = None
        current_support_act = None
        continue

    if day_number_pattern.match(line):
        current_day_number = line
        current_event_name = None
        current_support_act = None
        continue

    if line.startswith("W/ "):
        current_support_act = line.replace("W/ ", "").strip()
        continue

    time_match = time_pattern.match(line)

    if time_match:
        if current_month and current_year and current_day_number and current_event_name:

            date_text = f"{current_day_number} {current_month} {current_year}"
            event_time = time_match.group(1).replace(" ", "")

            events.append({
                "date_text": date_text,
                "day_of_week": current_day_name,
                "time": event_time,
                "event_name": current_event_name,
                "support_act": current_support_act,
                "venue": "Hordern Pavilion",
                "source": url
            })

        current_event_name = None
        current_support_act = None
        continue

    ignore_lines = [
        "Add to calendar",
        "More info",
        "Hordern Pavilion"
    ]

    if line not in ignore_lines and "Hordern Pavilion" not in line:
        current_event_name = line

if len(events) == 0:
    print("No events found.")
    print("Open hordern_page_text.txt to inspect the page text.")
else:
    df = pd.DataFrame(events)

    df["date"] = pd.to_datetime(
        df["date_text"],
        errors="coerce",
        dayfirst=True
    )

    df["datetime"] = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"],
        errors="coerce"
    )

    df = df[[
        "datetime",
        "date",
        "day_of_week",
        "time",
        "event_name",
        "support_act",
        "venue",
        "source"
    ]]

    print(df.head(30))
    print(f"\nFound {len(df)} upcoming Hordern events.")

    df.to_csv("hordern_sydneymusic_events.csv", index=False)

    conn = sqlite3.connect("events_database.db")

    df.to_sql(
        "hordern_sydneymusic_events",
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()

    print("\nSaved:")
    print("- hordern_sydneymusic_events.csv")
    print("- events_database.db")