import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re
import time
from datetime import datetime
from urllib.parse import urljoin, unquote
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# ============================================================
# SETTINGS
# ============================================================

START_YEAR = 2016
END_YEAR = 2019

venue_name = "Royal Randwick Racecourse"
event_type = "horse_racing"

DEFAULT_RANDWICK_ATTENDANCE_PROXY = 15000

calendar_base_url = "https://racing.racingnsw.com.au/FreeFields/Calendar_Meetings.aspx"

# IMPORTANT:
# This is the folder that StageMeeting.aspx lives inside.
freefields_base_url = "https://racing.racingnsw.com.au/FreeFields/"

csv_file = "royal_randwick_race_meetings_2016_2019_clean.csv"
db_file = "royal_randwick_race_meetings_database.db"
table_name = "royal_randwick_race_meetings_2016_2019_clean"

events = []


# ============================================================
# SELENIUM SETUP
# ============================================================

options = webdriver.ChromeOptions()

# Keep visible while testing.
# Once it works, you can uncomment this line:
# options.add_argument("--headless=new")

options.add_argument("--window-size=1400,1000")

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install()),
    options=options
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(text):
    text = text.replace("\xa0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def get_calendar_page(month, year):
    """
    Opens the Racing NSW race diary calendar for one month.
    """

    date_string = f"01/{month:02d}/{year}"

    params = {
        "date": date_string
    }

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    }

    print(f"\nOpening calendar for {date_string}")

    try:
        response = requests.get(
            calendar_base_url,
            params=params,
            headers=headers,
            timeout=30
        )

        if response.status_code != 200:
            print(f"Calendar failed for {date_string}: status {response.status_code}")
            return None

        return response.text

    except Exception as e:
        print(f"Calendar error for {date_string}: {e}")
        return None


def extract_date_from_key(href):
    """
    Extracts the meeting date from links like:

    StageMeeting.aspx?Key=2016Feb27,NSW,Royal%20Randwick
    Results.aspx?Key=2016Apr25,NSW,Royal%20Randwick

    Returns:
    2016-02-27
    """

    decoded_href = unquote(href)

    match = re.search(
        r"Key=(\d{4}[A-Za-z]{3}\d{1,2}),NSW,Royal\s+Randwick",
        decoded_href
    )

    if not match:
        return None

    raw_date = match.group(1)

    try:
        date_obj = datetime.strptime(raw_date, "%Y%b%d")
        return date_obj.strftime("%Y-%m-%d")
    except ValueError:
        return None


def find_royal_randwick_links(calendar_html):
    """
    Finds Royal Randwick result/stage links on the monthly calendar.
    """

    soup = BeautifulSoup(calendar_html, "html.parser")

    found_links = []

    for a in soup.find_all("a"):
        link_text = clean_text(a.get_text(" ", strip=True))
        href = a.get("href", "")

        if not href:
            continue

        decoded_href = unquote(href)

        # Only Royal Randwick links
        if "Royal Randwick" not in link_text and "Royal Randwick" not in decoded_href:
            continue

        # Only meeting/result links
        if "StageMeeting.aspx" not in href and "Results.aspx" not in href:
            continue

        date_clean = extract_date_from_key(href)

        if date_clean is None:
            continue

        # CRITICAL FIX:
        # Join relative links against /FreeFields/, not the domain root.
        full_url = urljoin(freefields_base_url, href)

        found_links.append({
            "date": date_clean,
            "url": full_url,
            "text": link_text
        })

    # Remove duplicates by date
    unique = {}

    for item in found_links:
        unique[item["date"]] = item

    links = list(unique.values())
    links = sorted(links, key=lambda x: x["date"])

    return links


def get_meeting_page_text(meeting_url):
    """
    Opens the result page in Selenium and returns page text.
    """

    print(f"Opening meeting page: {meeting_url}")

    try:
        driver.get(meeting_url)
        time.sleep(3)

        soup = BeautifulSoup(driver.page_source, "html.parser")
        page_text = soup.get_text(" ", strip=True)

        return clean_text(page_text)

    except Exception as e:
        print(f"Could not open meeting page: {e}")
        return ""


def extract_races_from_meeting(page_text):
    """
    Extracts race number, time, name and distance from headings like:

    Race 1 - 12:40PM ANZAC DAY HANDICAP (1400 METRES)
    Race 2 - 1:15PM 1ST LIGHT HORSE PLATE (1400 METRES)
    """

    pattern = re.compile(
        r"Race\s+(\d+)\s*-\s*"
        r"(\d{1,2}:\d{2}\s?(?:AM|PM))\s+"
        r"(.+?)\s*"
        r"\((\d+)\s+METRES\)",
        re.IGNORECASE
    )

    matches = pattern.findall(page_text)

    races = []

    for race_number, race_time, race_name, distance in matches:
        race_number = int(race_number)
        race_time = race_time.upper().replace(" ", "")
        race_name = clean_text(race_name)
        distance = int(distance)

        races.append({
            "race_number": race_number,
            "race_time": race_time,
            "race_name": race_name,
            "distance_metres": distance
        })

    # Remove duplicates by race number
    unique = {}

    for race in races:
        unique[race["race_number"]] = race

    races = list(unique.values())
    races = sorted(races, key=lambda x: x["race_number"])

    return races


def race_time_to_datetime(date_clean, race_time):
    race_time = race_time.upper().replace(" ", "")

    return pd.to_datetime(
        f"{date_clean} {race_time}",
        format="%Y-%m-%d %I:%M%p",
        errors="coerce"
    )


def build_event_from_meeting(date_clean, meeting_url, races):
    """
    Creates one event row for the full race meeting day.
    """

    if not races:
        return None

    first_race = races[0]
    last_race = races[-1]

    first_race_time = first_race["race_time"]
    last_race_time = last_race["race_time"]

    start = race_time_to_datetime(date_clean, first_race_time)
    end = race_time_to_datetime(date_clean, last_race_time)

    number_of_races = len(races)

    events_on_day = " | ".join([
        f"Race {race['race_number']}: {race['race_name']} ({race['distance_metres']}m)"
        for race in races
    ])

    event_name = f"Royal Randwick Race Meeting - {date_clean}"

    return {
        "event_name": event_name,
        "venue": venue_name,
        "event_type": event_type,
        "date": date_clean,
        "first_race_time": first_race_time,
        "last_race_time": last_race_time,
        "start": start,
        "end": end,
        "number_of_races": number_of_races,
        "events_on_day": events_on_day,
        "estimated_attendance": DEFAULT_RANDWICK_ATTENDANCE_PROXY,
        "attendance_source": "venue_capacity_proxy",
        "source_url": meeting_url
    }


# ============================================================
# MAIN SCRAPING LOOP
# ============================================================

seen_dates = set()

for year in range(START_YEAR, END_YEAR + 1):
    print("\n==============================")
    print(f"Searching Royal Randwick meetings for {year}")
    print("==============================")

    for month in range(1, 13):
        calendar_html = get_calendar_page(month, year)

        if calendar_html is None:
            continue

        randwick_links = find_royal_randwick_links(calendar_html)

        print(f"Found {len(randwick_links)} Royal Randwick links in {month:02d}/{year}")

        for item in randwick_links:
            date_clean = item["date"]
            meeting_url = item["url"]

            if date_clean in seen_dates:
                continue

            seen_dates.add(date_clean)

            page_text = get_meeting_page_text(meeting_url)

            # Debug check: see if page is still wrong
            if "The resource cannot be found" in page_text:
                print(f"404 page opened for {date_clean}, skipping.")
                print(meeting_url)
                continue

            races = extract_races_from_meeting(page_text)

            if not races:
                print(f"No race times found for {date_clean}")
                print(f"Checked URL: {meeting_url}")
                continue

            event = build_event_from_meeting(
                date_clean=date_clean,
                meeting_url=meeting_url,
                races=races
            )

            if event is not None:
                events.append(event)

                print(
                    f"FOUND {date_clean} | "
                    f"First: {event['first_race_time']} | "
                    f"Last: {event['last_race_time']} | "
                    f"Races: {event['number_of_races']}"
                )

            time.sleep(1)

        time.sleep(1)


driver.quit()


# ============================================================
# SAVE CLEAN CSV AND DATABASE
# ============================================================

df = pd.DataFrame(events)

if df.empty:
    print("\nNo Royal Randwick race meetings found.")
    print("This probably means the meeting page format is different from the expected Race 1 - 12:40PM format.")

else:
    # Remove duplicate meeting days
    df = df.drop_duplicates(
        subset=["date", "venue", "event_type"]
    )

    # Convert dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")

    # Add modelling columns
    df["year"] = df["start"].dt.year
    df["month"] = df["start"].dt.month
    df["day"] = df["start"].dt.day
    df["start_hour"] = df["start"].dt.hour
    df["end_hour"] = df["end"].dt.hour
    df["day_of_week"] = df["start"].dt.day_name()
    df["is_weekend"] = df["start"].dt.dayofweek.isin([5, 6]).astype(int)

    # Sort properly
    df = df.sort_values("start").reset_index(drop=True)

    # Add event ID
    df.insert(0, "event_id", range(1, len(df) + 1))

    # Final clean columns
    df = df[[
        "event_id",
        "event_name",
        "venue",
        "event_type",
        "date",
        "first_race_time",
        "last_race_time",
        "start",
        "end",
        "number_of_races",
        "events_on_day",
        "estimated_attendance",
        "attendance_source",
        "year",
        "month",
        "day",
        "day_of_week",
        "start_hour",
        "end_hour",
        "is_weekend",
        "source_url"
    ]]

    # Convert datetime columns to CSV-friendly strings
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["start"] = df["start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["end"] = df["end"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Save CSV
    df.to_csv(
        csv_file,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n"
    )

    # Save SQLite database
    conn = sqlite3.connect(db_file)

    df.to_sql(
        table_name,
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()

    # Test-read CSV
    test_df = pd.read_csv(csv_file)

    print("\nCSV saved successfully.")
    print(f"Rows saved: {len(test_df)}")
    print(f"Columns saved: {list(test_df.columns)}")

    print("\nPreview:")
    print(test_df.head(20))

    print("\nRace meetings by year:")
    print(test_df["year"].value_counts().sort_index())

    print("\nSaved files:")
    print(f"- {csv_file}")
    print(f"- {db_file}")
    print(f"- Table name: {table_name}")