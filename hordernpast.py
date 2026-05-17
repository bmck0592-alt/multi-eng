from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re
import time

base_url = "https://www.setlist.fm/venue/hordern-pavilion-sydney-australia-73d63e0d.html"

years_to_scrape = list(range(2013, 2020))

# Use this as a modelling proxy, not confirmed per-event attendance
DEFAULT_HORDERN_ATTENDANCE_PROXY = 5500

month_map = {
    "JAN": "January",
    "FEB": "February",
    "MAR": "March",
    "APR": "April",
    "MAY": "May",
    "JUN": "June",
    "JUL": "July",
    "AUG": "August",
    "SEP": "September",
    "OCT": "October",
    "NOV": "November",
    "DEC": "December",
    "Jan": "January",
    "Feb": "February",
    "Mar": "March",
    "Apr": "April",
    "May": "May",
    "Jun": "June",
    "Jul": "July",
    "Aug": "August",
    "Sep": "September",
    "Oct": "October",
    "Nov": "November",
    "Dec": "December",
}

events = []

driver = webdriver.Chrome(
    service=Service(ChromeDriverManager().install())
)


def find_year_dropdown():
    selects = driver.find_elements(By.TAG_NAME, "select")

    for select_element in selects:
        try:
            select = Select(select_element)
            option_texts = [option.text.strip() for option in select.options]

            if any("2019" in text for text in option_texts):
                return select

        except Exception:
            continue

    return None


def select_year(year):
    driver.get(base_url)
    time.sleep(4)

    year_dropdown = find_year_dropdown()

    if year_dropdown is None:
        print("Could not find year dropdown.")
        return False

    matched_option = None

    for option in year_dropdown.options:
        option_text = option.text.strip()

        if option_text.startswith(str(year)):
            matched_option = option_text
            break

    if matched_option is None:
        print(f"Year {year} not found in dropdown.")
        return False

    print(f"Selecting year: {matched_option}")

    year_dropdown.select_by_visible_text(matched_option)
    time.sleep(5)

    return True


def get_page_urls_for_current_year():
    urls = set()
    urls.add(driver.current_url)

    links = driver.find_elements(By.TAG_NAME, "a")

    for link in links:
        href = link.get_attribute("href")

        if not href:
            continue

        if "hordern-pavilion-sydney-australia-73d63e0d" in href and "year=" in href:
            urls.add(href)

    return sorted(urls)


def extract_time(text):
    text = text.replace("\xa0", " ")

    scheduled_match = re.search(
        r"Scheduled:\s*(\d{1,2}:\d{2}\s?(?:AM|PM|am|pm))",
        text
    )

    if scheduled_match:
        return scheduled_match.group(1).upper().replace(" ", " ")

    doors_match = re.search(
        r"Doors\s*(\d{1,2}:\d{2}\s?(?:AM|PM|am|pm))",
        text
    )

    if doors_match:
        return doors_match.group(1).upper().replace(" ", " ")

    any_time = re.search(
        r"\b(\d{1,2}:\d{2}\s?(?:AM|PM|am|pm))\b",
        text
    )

    if any_time:
        return any_time.group(1).upper().replace(" ", " ")

    return "7:00 PM"


def parse_current_page(selected_year):
    soup = BeautifulSoup(driver.page_source, "html.parser")
    page_text = soup.get_text("\n", strip=True).replace("\xa0", " ")

    lines = [line.strip() for line in page_text.split("\n") if line.strip()]

    page_events = []

    i = 0

    while i < len(lines) - 3:
        month_line = lines[i]
        day_line = lines[i + 1]
        year_line = lines[i + 2]

        if (
            month_line in month_map
            and day_line.isdigit()
            and year_line == str(selected_year)
        ):
            date_text = f"{day_line} {month_map[month_line]} {selected_year}"

            # Look forward through this event block
            block_lines = lines[i:i + 80]
            block_text = "\n".join(block_lines)

            event_name = None

            for line in block_lines:
                # Normal format:
                # Artist at Hordern Pavilion, Sydney, Australia
                match = re.search(
                    r"^(.*?)\s+at\s+Hordern Pavilion,\s+Sydney,\s+Australia",
                    line
                )

                if match:
                    event_name = match.group(1).strip()
                    break

                # Festival / named event format:
                # The Script at Optus RockCorps 2013
                if " at " in line and "Hordern Pavilion" not in line:
                    possible_name = line.split(" at ")[0].strip()

                    if possible_name and len(possible_name) > 1:
                        event_name = possible_name
                        break

            if event_name:
                event_time = extract_time(block_text)

                page_events.append({
                    "date_text": date_text,
                    "time": event_time,
                    "event_name": event_name,
                    "venue": "Hordern Pavilion",
                    "event_type": "concert",
                    "estimated_attendance": DEFAULT_HORDERN_ATTENDANCE_PROXY,
                    "attendance_source": "venue_capacity_proxy",
                    "source_url": driver.current_url
                })

            i += 3

        else:
            i += 1

    return page_events


# ==============================
# Main scraping loop
# ==============================

for year in years_to_scrape:
    print("\n==============================")
    print(f"Scraping year {year}")
    print("==============================")

    if not select_year(year):
        continue

    page_urls = get_page_urls_for_current_year()

    print(f"Found {len(page_urls)} pages for {year}")

    year_total = 0

    for page_number, page_url in enumerate(page_urls, start=1):
        print(f"Scraping {year}, page {page_number}: {page_url}")

        driver.get(page_url)
        time.sleep(3)

        page_events = parse_current_page(year)

        print(f"Found {len(page_events)} events on this page")

        year_total += len(page_events)
        events.extend(page_events)

    print(f"Total found for {year}: {year_total}")

driver.quit()


# ==============================
# Save clean CSV and database
# ==============================

df = pd.DataFrame(events)

if df.empty:
    print("\nNo Hordern Pavilion events found.")

else:
    # Remove duplicate events
    df = df.drop_duplicates(
        subset=["date_text", "event_name", "venue"]
    )

    # Convert date text into datetime
    df["date"] = pd.to_datetime(
        df["date_text"],
        errors="coerce",
        dayfirst=True
    )

    # Combine date and time into start datetime
    df["start"] = pd.to_datetime(
        df["date"].dt.strftime("%Y-%m-%d") + " " + df["time"],
        errors="coerce"
    )

    # Assume each event lasts 4 hours
    df["end"] = df["start"] + pd.Timedelta(hours=4)

    # Add modelling columns
    df["year"] = df["start"].dt.year
    df["month"] = df["start"].dt.month
    df["day"] = df["start"].dt.day
    df["hour"] = df["start"].dt.hour
    df["day_of_week"] = df["start"].dt.day_name()
    df["is_weekend"] = df["start"].dt.dayofweek.isin([5, 6]).astype(int)

    # Sort by time
    df = df.sort_values("start").reset_index(drop=True)

    # Add event ID
    df.insert(0, "event_id", range(1, len(df) + 1))

    # Keep final clean columns only
    df = df[[
        "event_id",
        "event_name",
        "venue",
        "event_type",
        "date",
        "time",
        "start",
        "end",
        "estimated_attendance",
        "attendance_source",
        "year",
        "month",
        "day",
        "day_of_week",
        "hour",
        "is_weekend",
        "source_url"
    ]]

    # Convert datetime columns to CSV-friendly text
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["start"] = df["start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["end"] = df["end"].dt.strftime("%Y-%m-%d %H:%M:%S")

    # Output filenames
    csv_file = "hordern_events_2013_2019_clean.csv"
    db_file = "events_database.db"
    table_name = "hordern_events_2013_2019_clean"

    # Save as a proper CSV
    df.to_csv(
        csv_file,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n"
    )

    # Save to SQLite database
    conn = sqlite3.connect(db_file)

    df.to_sql(
        table_name,
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()

    # Test-read the CSV back in to confirm it saved properly
    test_df = pd.read_csv(csv_file)

    print("\nCSV saved successfully.")
    print(f"Rows saved: {len(test_df)}")
    print(f"Columns saved: {list(test_df.columns)}")

    print("\nPreview of CSV as dataframe:")
    print(test_df.head(10))

    print("\nEvents by year:")
    print(test_df["year"].value_counts().sort_index())

    print("\nSaved files:")
    print(f"- {csv_file}")
    print(f"- {db_file}")
    print(f"- Table name: {table_name}")