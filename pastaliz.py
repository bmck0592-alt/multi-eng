import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re
from datetime import datetime
from urllib.parse import urljoin

# ============================================================
# SETTINGS
# ============================================================

START_YEAR = 2013
END_YEAR = 2019

VENUE_NAME = "Sydney Football Stadium / Allianz Stadium"

CSV_FILE = "allianz_sfs_events_2013_2019_clean.csv"
DB_FILE = "allianz_sfs_events_2013_2019_database.db"
TABLE_NAME = "allianz_sfs_events_2013_2019"

ULTIMATE_ALEAGUE_URL = "https://www.ultimatealeague.com/stadium/?stadium_id=7"

RLP_BASE_URL = "https://www.rugbyleagueproject.org/venues/sydney-football-stadium--old-/results.html"

SETLIST_URLS = [
    "https://www.setlist.fm/venue/allianz-stadium-sydney-australia-6bd4f23e.html",
    "https://www.setlist.fm/venue/sydney-football-stadium-sydney-australia-3d62143.html",
    "https://www.setlist.fm/venue/aussie-stadium-sydney-australia-1bd30dbc.html",
]

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

events = []


# ============================================================
# GENERAL HELPERS
# ============================================================

def clean_text(x):
    if x is None:
        return ""

    x = str(x)
    x = x.replace("\xa0", " ")
    x = re.sub(r"\s+", " ", x)

    return x.strip()


def parse_int(x):
    x = clean_text(x)

    if not x or x.lower() in ["nan", "none", "-"]:
        return None

    x = x.replace(",", "")

    match = re.search(r"\d+", x)

    if not match:
        return None

    return int(match.group(0))


def parse_date_flexible(x, default_year=None):
    x = clean_text(x)

    if not x:
        return None

    # Example: "Mar 28" + 2014
    if default_year and re.match(r"^[A-Za-z]{3}\s+\d{1,2}$", x):
        x = f"{x} {default_year}"

    # Example: "28 Mar" + 2014
    if default_year and re.match(r"^\d{1,2}\s+[A-Za-z]{3}$", x):
        x = f"{x} {default_year}"

    formats = [
        "%d/%m/%Y",
        "%d/%m/%y",
        "%a %d/%m/%Y",
        "%a %d/%m/%y",
        "%d-%b-%Y",
        "%d %b %Y",
        "%d %B %Y",
        "%b %d %Y",
        "%B %d %Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            continue

    return None


def in_range(date_obj):
    return date_obj is not None and START_YEAR <= date_obj.year <= END_YEAR


def add_event(
    event_name,
    event_type,
    event_category,
    sport_or_category,
    date_obj,
    source,
    source_url,
    attendance=None,
    attendance_source=None,
    start_time=None,
    home_team=None,
    away_team=None,
    home_score=None,
    away_score=None,
    competition=None,
    round_name=None,
    result=None,
    notes=None
):
    if not in_range(date_obj):
        return

    date_clean = date_obj.strftime("%Y-%m-%d")

    start = pd.NaT
    end = pd.NaT

    if start_time:
        start = pd.to_datetime(f"{date_clean} {start_time}", errors="coerce")

        if pd.notna(start):
            duration_hours = 4 if event_category == "concert" else 2.5
            end = start + pd.Timedelta(hours=duration_hours)

    events.append({
        "event_name": clean_text(event_name),
        "venue": VENUE_NAME,
        "event_type": clean_text(event_type),
        "event_category": clean_text(event_category),
        "sport_or_category": clean_text(sport_or_category),
        "date": date_clean,
        "start_time": start_time,
        "end_time": None,
        "start": start,
        "end": end,
        "attendance": attendance,
        "attendance_source": attendance_source,
        "source": source,
        "source_url": source_url,
        "competition": clean_text(competition),
        "round": clean_text(round_name),
        "home_team": clean_text(home_team),
        "away_team": clean_text(away_team),
        "home_score": home_score,
        "away_score": away_score,
        "result": clean_text(result),
        "notes": notes,
    })


# ============================================================
# 1. ULTIMATE A-LEAGUE SCRAPER
# ============================================================

def scrape_ultimate_aleague():
    print("\n==============================")
    print("Scraping Ultimate A-League")
    print("==============================")

    try:
        response = requests.get(ULTIMATE_ALEAGUE_URL, headers=HEADERS, timeout=30)
        response.raise_for_status()
    except Exception as e:
        print(f"Ultimate A-League failed: {e}")
        return

    soup = BeautifulSoup(response.text, "html.parser")
    page_text = clean_text(soup.get_text(" ", strip=True))

    pattern = re.compile(
        r"(?P<season>\d{4}-\d{2})\s+"
        r"(?P<round>[A-Za-z0-9]+)\s+"
        r"(?P<date>[A-Za-z]{3}\s+\d{2}/\d{2}/\d{4})\s+"
        r"(?P<crowd>\d{1,3}(?:,\d{3})*)\s+"
        r"(?P<home>.+?)\s+"
        r"(?P<home_score>\d+)\s+-\s+(?P<away_score>\d+)\s+"
        r"(?P<away>.+?)"
        r"(?=\s+\d{4}-\d{2}\s+[A-Za-z0-9]+\s+[A-Za-z]{3}\s+\d{2}/\d{2}/\d{4}|$)"
    )

    added = 0

    for match in pattern.finditer(page_text):
        date_obj = parse_date_flexible(match.group("date"))

        if not in_range(date_obj):
            continue

        home_team = clean_text(match.group("home"))
        away_team = clean_text(match.group("away"))

        # Fix duplicated club-code/score text.
        away_team = re.sub(r"\s+[A-Z]{2,4}\s+\d+-\d+$", "", away_team).strip()
        away_team = re.sub(r"\s+\d+-\d+$", "", away_team).strip()

        home_score = parse_int(match.group("home_score"))
        away_score = parse_int(match.group("away_score"))
        crowd = parse_int(match.group("crowd"))

        result = f"{home_team} {home_score} - {away_score} {away_team}"

        add_event(
            event_name=f"A-League: {home_team} v {away_team}",
            event_type="a_league",
            event_category="sport",
            sport_or_category="football",
            date_obj=date_obj,
            source="Ultimate A-League",
            source_url=ULTIMATE_ALEAGUE_URL,
            attendance=crowd,
            attendance_source="reported_crowd" if crowd else None,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            competition=match.group("season"),
            round_name=match.group("round"),
            result=result,
            notes="Ultimate A-League stadium_id=7; kick-off time not available from summary table"
        )

        added += 1

    print(f"A-League rows added: {added}")


# ============================================================
# 2. RUGBY LEAGUE PROJECT SCRAPER
# ============================================================

def get_rlp_page_urls(max_pages=35):
    """
    Rugby League Project has paginated results.
    We generate page URLs directly because relying on page links can miss pages
    or pick up unrelated archive links.
    """

    urls = [RLP_BASE_URL]

    for page in range(2, max_pages + 1):
        urls.append(f"{RLP_BASE_URL}?page={page}")

    return urls


def extract_year_from_competition(competition):
    competition = clean_text(competition)

    match = re.search(r"(20\d{2}|19\d{2})", competition)

    if not match:
        return None

    return int(match.group(1))


def parse_rlp_date(date_text, current_month, current_year):
    """
    RLP date cells may appear as:
    - "Mar 28"
    - "Apr 4"
    - "28"
    - "Fri Mar 28" depending on page format

    This carries the month forward where only a day number is shown.
    """

    date_text = clean_text(date_text)

    if not date_text:
        return None, current_month

    # Remove weekday if present.
    date_text = re.sub(r"^(Mon|Tue|Wed|Thu|Fri|Sat|Sun)\s+", "", date_text)

    # Case: "Mar 28"
    month_day = re.match(r"^([A-Za-z]{3})\s+(\d{1,2})$", date_text)
    if month_day:
        month = month_day.group(1)
        day = month_day.group(2)
        date_obj = parse_date_flexible(f"{month} {day} {current_year}")
        return date_obj, month

    # Case: "28 Mar"
    day_month = re.match(r"^(\d{1,2})\s+([A-Za-z]{3})$", date_text)
    if day_month:
        day = day_month.group(1)
        month = day_month.group(2)
        date_obj = parse_date_flexible(f"{day} {month} {current_year}")
        return date_obj, month

    # Case: only "28"
    just_day = re.match(r"^\d{1,2}$", date_text)
    if just_day and current_month:
        date_obj = parse_date_flexible(f"{current_month} {date_text} {current_year}")
        return date_obj, current_month

    # Case: full date.
    date_obj = parse_date_flexible(date_text, default_year=current_year)

    return date_obj, current_month


def find_rlp_result_tables(soup):
    """
    Finds tables that look like Rugby League Project result tables.
    """

    tables = []

    for table in soup.find_all("table"):
        table_text = clean_text(table.get_text(" ", strip=True)).lower()

        has_competition = "competition" in table_text
        has_round = "round" in table_text
        has_home = "home" in table_text
        has_away = "away" in table_text
        has_venue = "venue" in table_text

        if has_competition and has_round and has_home and has_away and has_venue:
            tables.append(table)

    return tables


def scrape_rugby_league_project():
    print("\n==============================")
    print("Scraping Rugby League Project")
    print("==============================")

    urls = get_rlp_page_urls(max_pages=35)

    valid_venues = [
        "Sydney Football Stadium",
        "Allianz Stadium",
        "Aussie Stadium",
    ]

    total_added = 0
    pages_with_rows = 0
    empty_pages_in_a_row = 0

    for url in urls:
        print(f"Parsing RLP page: {url}")

        try:
            response = requests.get(url, headers=HEADERS, timeout=30)

            if response.status_code == 404:
                print("Page not found, stopping RLP pagination.")
                break

            response.raise_for_status()

        except Exception as e:
            print(f"Failed RLP page: {url} | {e}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")

        tables = find_rlp_result_tables(soup)

        # If no exact result table is found, fall back to all tables.
        if not tables:
            tables = soup.find_all("table")

        page_added = 0

        for table in tables:
            rows = table.find_all("tr")

            current_month_by_year = {}

            for row in rows:
                cells = row.find_all(["td", "th"])

                if len(cells) < 8:
                    continue

                texts = [clean_text(cell.get_text(" ", strip=True)) for cell in cells]
                row_text = clean_text(" ".join(texts))

                # Skip headers.
                if texts[0].lower() in ["competition", "comp"]:
                    continue

                # Important fix: 2013-2019 often says Allianz Stadium.
                if not any(venue in row_text for venue in valid_venues):
                    continue

                # Common layout:
                # 0 Competition
                # 1 Round
                # 2 Date
                # 3 Home
                # 4 Home score
                # 5 Away
                # 6 Away score
                # 7 Venue
                # 8 Crowd
                competition = texts[0] if len(texts) > 0 else None
                round_name = texts[1] if len(texts) > 1 else None
                date_text = texts[2] if len(texts) > 2 else None
                home_team = texts[3] if len(texts) > 3 else None
                home_score = parse_int(texts[4]) if len(texts) > 4 else None
                away_team = texts[5] if len(texts) > 5 else None
                away_score = parse_int(texts[6]) if len(texts) > 6 else None
                listed_venue = texts[7] if len(texts) > 7 else None
                crowd = parse_int(texts[8]) if len(texts) > 8 else None

                year = extract_year_from_competition(competition)

                if year is None:
                    continue

                if year < START_YEAR or year > END_YEAR:
                    continue

                current_month = current_month_by_year.get(year)

                date_obj, updated_month = parse_rlp_date(
                    date_text=date_text,
                    current_month=current_month,
                    current_year=year
                )

                if updated_month:
                    current_month_by_year[year] = updated_month

                if not in_range(date_obj):
                    continue

                detail_url = url
                links = row.find_all("a", href=True)

                if links:
                    detail_url = urljoin(url, links[-1]["href"])

                result = None
                if home_team and away_team and home_score is not None and away_score is not None:
                    result = f"{home_team} {home_score} - {away_score} {away_team}"

                add_event(
                    event_name=f"Rugby League: {home_team} v {away_team}",
                    event_type="rugby_league",
                    event_category="sport",
                    sport_or_category="rugby league",
                    date_obj=date_obj,
                    source="Rugby League Project",
                    source_url=detail_url,
                    attendance=crowd,
                    attendance_source="reported_crowd" if crowd else None,
                    home_team=home_team,
                    away_team=away_team,
                    home_score=home_score,
                    away_score=away_score,
                    competition=competition,
                    round_name=round_name,
                    result=result,
                    notes=f"Rugby League Project venue row; listed venue = {listed_venue}"
                )

                page_added += 1
                total_added += 1

        print(f"RLP rows from this page: {page_added}")

        if page_added > 0:
            pages_with_rows += 1
            empty_pages_in_a_row = 0
        else:
            empty_pages_in_a_row += 1

        # Stop after a lot of empty pages once we have already found useful pages.
        if pages_with_rows > 0 and empty_pages_in_a_row >= 8:
            print("Stopping RLP after 8 empty pages in a row.")
            break

    print(f"Rugby League rows added: {total_added}")


# ============================================================
# 3. SETLIST.FM CONCERT SCRAPER
# ============================================================

def scrape_single_setlist_url(base_url, max_pages=10):
    added = 0

    venue_pattern = re.compile(
        r"^(.*?)\s+at\s+"
        r"(Allianz Stadium|Sydney Football Stadium|Aussie Stadium),\s+"
        r"Sydney,\s+Australia",
        re.I
    )

    date_pattern = re.compile(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+"
        r"\d{1,2}\s+"
        r"\d{4}\b"
    )

    time_pattern = re.compile(
        r"\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\b"
    )

    for page in range(1, max_pages + 1):
        url = base_url if page == 1 else f"{base_url}?page={page}"

        try:
            response = requests.get(url, headers=HEADERS, timeout=30)

            if response.status_code == 404:
                continue

            response.raise_for_status()

        except Exception:
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        lines = [
            clean_text(line)
            for line in soup.get_text("\n").split("\n")
            if clean_text(line)
        ]

        page_added = 0

        for i, line in enumerate(lines):
            match = venue_pattern.search(line)

            if not match:
                continue

            artist = clean_text(match.group(1))
            nearby = " ".join(lines[max(0, i - 10):i + 50])

            date_match = date_pattern.search(nearby)

            if not date_match:
                continue

            date_obj = parse_date_flexible(date_match.group(0))

            if not in_range(date_obj):
                continue

            time_match = time_pattern.search(nearby)
            start_time = None

            if time_match:
                start_time = time_match.group(1).upper().replace(" ", "")

            add_event(
                event_name=artist,
                event_type="concert",
                event_category="concert",
                sport_or_category="concert/performance",
                date_obj=date_obj,
                source="setlist.fm",
                source_url=url,
                attendance=None,
                attendance_source=None,
                start_time=start_time,
                notes="setlist.fm venue listing; attendance usually unavailable"
            )

            page_added += 1
            added += 1

        if page_added == 0 and page > 3:
            # Avoid wasting time on lots of empty pages.
            continue

    return added


def scrape_setlists():
    print("\n==============================")
    print("Scraping setlist.fm venue aliases")
    print("==============================")

    total_added = 0

    for url in SETLIST_URLS:
        count = scrape_single_setlist_url(url, max_pages=10)
        print(f"{url} -> {count} rows")
        total_added += count

    print(f"setlist.fm rows added: {total_added}")


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_outputs():
    df = pd.DataFrame(events)

    if df.empty:
        print("\nNo events found.")
        return

    # Basic cleaning
    df = df[df["event_name"].notna()]
    df = df[df["event_name"].astype(str).str.strip() != ""]

    bad_event_names = [
        "uploaded by",
        "embedded by",
        "setlists",
        "videos",
        "photos",
        "show duplicate",
        "show duplicates",
    ]

    df = df[~df["event_name"].str.lower().isin(bad_event_names)]

    # Date processing
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df = df[
        (df["date"].dt.year >= START_YEAR) &
        (df["date"].dt.year <= END_YEAR)
    ]

    # Remove duplicated rows across sources / venue aliases
    df = df.drop_duplicates(
        subset=["event_name", "venue", "date", "event_type"],
        keep="first"
    )

    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")

    # Modelling columns
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    df["start_hour"] = df["start"].dt.hour
    df["end_hour"] = df["end"].dt.hour

    # Sort and add ID
    df = df.sort_values(["date", "event_type", "event_name"]).reset_index(drop=True)
    df.insert(0, "event_id", range(1, len(df) + 1))

    # CSV-friendly formatting
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    df["start"] = df["start"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df["end"] = df["end"].dt.strftime("%Y-%m-%d %H:%M:%S")

    final_cols = [
        "event_id",
        "event_name",
        "venue",
        "event_type",
        "event_category",
        "sport_or_category",
        "date",
        "start_time",
        "end_time",
        "start",
        "end",
        "attendance",
        "attendance_source",
        "source",
        "competition",
        "round",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "result",
        "year",
        "month",
        "day",
        "day_of_week",
        "start_hour",
        "end_hour",
        "is_weekend",
        "source_url",
        "notes",
    ]

    for col in final_cols:
        if col not in df.columns:
            df[col] = None

    df = df[final_cols]

    # Save CSV
    df.to_csv(
        CSV_FILE,
        index=False,
        encoding="utf-8-sig",
        lineterminator="\n"
    )

    # Save SQLite
    conn = sqlite3.connect(DB_FILE)

    df.to_sql(
        TABLE_NAME,
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()

    print("\n==============================")
    print("Saved Allianz/SFS database")
    print("==============================")

    print(f"Rows saved: {len(df)}")
    print(f"CSV: {CSV_FILE}")
    print(f"DB: {DB_FILE}")
    print(f"Table: {TABLE_NAME}")

    print("\nRows by source:")
    print(df["source"].value_counts())

    print("\nRows by event type:")
    print(df["event_type"].value_counts())

    print("\nMissing dates:")
    print(df["date"].isna().sum())

    print("\nPreview:")
    print(df.head(40))


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    scrape_ultimate_aleague()
    scrape_rugby_league_project()
    scrape_setlists()
    save_outputs()