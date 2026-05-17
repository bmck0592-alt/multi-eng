import requests
from bs4 import BeautifulSoup
import pandas as pd
import sqlite3
import re
import zipfile
import io
import json
from datetime import datetime

# ============================================================
# SETTINGS
# ============================================================

VENUE_NAME = "Sydney Cricket Ground"
START_YEAR = 2013
END_YEAR = 2019

CSV_FILE = "scg_events_2013_2019_clean.csv"
DB_FILE = "scg_events_2013_2019_database.db"
TABLE_NAME = "scg_events_2013_2019"

AFL_TABLES_URL = "https://afltables.com/afl/venues/scg_gm.html"
ULTIMATE_ALEAGUE_SCG_URL = "https://www.ultimatealeague.com/stadium/?stadium_id=37"
CONCERT_ARCHIVES_URL = "https://www.concertarchives.org/venues/sydney-cricket-ground"
SETLIST_BASE_URL = "https://www.setlist.fm/venue/sydney-cricket-ground-sydney-australia-63d626d7.html"

CRICSHEET_ZIPS = {
    "tests": "https://cricsheet.org/downloads/tests_json.zip",
    "odis": "https://cricsheet.org/downloads/odis_json.zip",
    "t20s": "https://cricsheet.org/downloads/t20s_json.zip",
    "bbl": "https://cricsheet.org/downloads/bbl_json.zip",
}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0 Safari/537.36"
    )
}

events = []


# ============================================================
# HELPERS
# ============================================================

def clean_text(x):
    if x is None:
        return ""
    x = str(x).replace("\xa0", " ")
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def parse_int(x):
    x = clean_text(x)

    if not x:
        return None

    x = x.replace(",", "")
    match = re.search(r"\d+", x)

    return int(match.group(0)) if match else None


def parse_date_flexible(x):
    x = clean_text(x)

    formats = [
        "%d-%b-%Y",
        "%d/%m/%Y",
        "%a %d/%m/%Y",
        "%b %d, %Y",
        "%B %d, %Y",
        "%b %d %Y",
        "%B %d %Y",
        "%Y-%m-%d",
    ]

    for fmt in formats:
        try:
            return datetime.strptime(x, fmt)
        except ValueError:
            continue

    return None


def in_year_range(date_obj):
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
    notes=None
):
    if not in_year_range(date_obj):
        return

    date_clean = date_obj.strftime("%Y-%m-%d")

    start = pd.NaT
    end = pd.NaT

    if start_time:
        start = pd.to_datetime(f"{date_clean} {start_time}", errors="coerce")
        if pd.notna(start):
            duration = 4 if event_category == "concert" else 2.5
            end = start + pd.Timedelta(hours=duration)

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
        "home_team": clean_text(home_team),
        "away_team": clean_text(away_team),
        "home_score": home_score,
        "away_score": away_score,
        "competition": clean_text(competition),
        "round": clean_text(round_name),
        "notes": notes,
    })


# ============================================================
# 1. AFL TABLES - FIXED SCG PARSER
# ============================================================

def scrape_afl_tables_scg():
    print("\nScraping AFL Tables SCG games...")

    try:
        r = requests.get(AFL_TABLES_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"AFL Tables failed: {e}")
        return

    soup = BeautifulSoup(r.text, "html.parser")

    # This is the key fix:
    # Use newlines so each AFL game stays as a separate row.
    lines = [
        clean_text(line)
        for line in soup.get_text("\n", strip=True).split("\n")
        if clean_text(line)
    ]

    added = 0
    debug_matched_any = False

    for line in lines:
        # Only actual game rows start like:
        # 411 R23,2019 ...
        if not re.match(r"^\d+\s+(R\d+|EF|QF|SF|PF|GF),\d{4}\s+", line):
            continue

        # Remove quarter-by-quarter score blocks inside backticks.
        # Example:
        # Sydney`6.2 6.3 11.4 17.7 ` 109 St Kilda`...` 64 33,722 24-Aug-2019
        line_clean = re.sub(r"`[^`]*`", " ", line)
        line_clean = clean_text(line_clean)

        # Now line looks like:
        # 411 R23,2019 Sydney 109 St Kilda 64 33,722 24-Aug-2019

        pattern = re.compile(
            r"^(?P<game_no>\d+)\s+"
            r"(?P<round>R\d+|EF|QF|SF|PF|GF),"
            r"(?P<year>\d{4})\s+"
            r"(?P<body>.+?)\s+"
            r"(?P<crowd>\d{1,3}(?:,\d{3})*)\s+"
            r"(?P<date>\d{2}-[A-Za-z]{3}-\d{4})$"
        )

        m = pattern.search(line_clean)

        if not m:
            continue

        debug_matched_any = True

        date_obj = parse_date_flexible(m.group("date"))

        if not in_year_range(date_obj):
            continue

        body = clean_text(m.group("body"))

        # Split body from the right:
        # team1 score1 team2 score2
        body_match = re.match(
            r"(?P<team1>.+?)\s+"
            r"(?P<score1>\d+)\s+"
            r"(?P<team2>.+?)\s+"
            r"(?P<score2>\d+)$",
            body
        )

        if not body_match:
            print(f"Could not parse AFL body: {body}")
            continue

        team1 = clean_text(body_match.group("team1"))
        team2 = clean_text(body_match.group("team2"))
        score1 = parse_int(body_match.group("score1"))
        score2 = parse_int(body_match.group("score2"))
        crowd = parse_int(m.group("crowd"))

        add_event(
            event_name=f"AFL: {team1} v {team2}",
            event_type="afl",
            event_category="sport",
            sport_or_category="Australian rules football",
            date_obj=date_obj,
            source="AFL Tables",
            source_url=AFL_TABLES_URL,
            attendance=crowd,
            attendance_source="reported_crowd",
            home_team=team1,
            away_team=team2,
            home_score=score1,
            away_score=score2,
            competition=m.group("year"),
            round_name=m.group("round"),
            notes="AFL Tables SCG all-games page"
        )

        added += 1

    if not debug_matched_any:
        print("WARNING: AFL parser did not match any AFL rows at all.")
        print("Printing first 20 likely AFL lines for debugging:")
        likely_lines = [
            line for line in lines
            if re.search(r"\b(R\d+|EF|QF|SF|PF|GF),\d{4}\b", line)
        ]
        for line in likely_lines[:20]:
            print(line)

    print(f"AFL rows added: {added}")


# ============================================================
# 2. ULTIMATE A-LEAGUE - SCG MATCHES
# ============================================================

def scrape_ultimate_aleague_scg():
    print("\nScraping Ultimate A-League SCG matches...")

    try:
        r = requests.get(ULTIMATE_ALEAGUE_SCG_URL, headers=HEADERS, timeout=30)
        r.raise_for_status()
    except Exception as e:
        print(f"Ultimate A-League failed: {e}")
        return

    soup = BeautifulSoup(r.text, "html.parser")
    text = clean_text(soup.get_text(" ", strip=True))

    pattern = re.compile(
        r"(?P<season>\d{4}-\d{2})\s+"
        r"(?P<round>[A-Za-z0-9]+)\s+"
        r"(?P<date>[A-Za-z]{3}\s+\d{2}/\d{2}/\d{4})\s+"
        r"(?P<crowd>\d{1,3}(?:,\d{3})*)\s+"
        r"(?P<home>[A-Za-z .'-]+?)\s+"
        r"(?P<home_score>\d+)\s+-\s+(?P<away_score>\d+)\s+"
        r"(?P<away>[A-Za-z .'-]+?)"
        r"(?=\s+[A-Z]{2,4}\s+\d+-\d+|\s+\d{4}-\d{2}|$)"
    )

    added = 0

    for m in pattern.finditer(text):
        date_obj = parse_date_flexible(m.group("date"))

        if not in_year_range(date_obj):
            continue

        home = clean_text(m.group("home"))
        away = clean_text(m.group("away"))

        add_event(
            event_name=f"A-League: {home} v {away}",
            event_type="a_league",
            event_category="sport",
            sport_or_category="football",
            date_obj=date_obj,
            source="Ultimate A-League",
            source_url=ULTIMATE_ALEAGUE_SCG_URL,
            attendance=parse_int(m.group("crowd")),
            attendance_source="reported_crowd",
            home_team=home,
            away_team=away,
            home_score=parse_int(m.group("home_score")),
            away_score=parse_int(m.group("away_score")),
            competition=m.group("season"),
            round_name=m.group("round"),
            notes="Ultimate A-League SCG stadium page"
        )

        added += 1

    print(f"A-League rows added: {added}")


# ============================================================
# 3. CRICSHEET - SCG CRICKET DATES
# ============================================================

def scrape_cricsheet_scg():
    print("\nScraping Cricsheet cricket matches at SCG...")

    total_added = 0

    for competition_name, zip_url in CRICSHEET_ZIPS.items():
        print(f"Downloading {competition_name}...")

        try:
            r = requests.get(zip_url, headers=HEADERS, timeout=60)
            r.raise_for_status()
        except Exception as e:
            print(f"Could not download {competition_name}: {e}")
            continue

        try:
            z = zipfile.ZipFile(io.BytesIO(r.content))
        except Exception as e:
            print(f"Could not open zip for {competition_name}: {e}")
            continue

        added_this_zip = 0

        for filename in z.namelist():
            if not filename.endswith(".json"):
                continue

            try:
                match = json.loads(z.read(filename).decode("utf-8"))
            except Exception:
                continue

            info = match.get("info", {})

            venue = clean_text(info.get("venue", ""))
            if venue.lower() != "sydney cricket ground":
                continue

            dates = info.get("dates", [])
            if not dates:
                continue

            date_obj = parse_date_flexible(str(dates[0]))
            if not in_year_range(date_obj):
                continue

            teams = info.get("teams", [])
            team1 = teams[0] if len(teams) > 0 else None
            team2 = teams[1] if len(teams) > 1 else None

            event = info.get("event", {})
            event_name_raw = event.get("name") if isinstance(event, dict) else None

            match_type = info.get("match_type", competition_name)

            if team1 and team2:
                event_name = f"Cricket: {team1} v {team2}"
            else:
                event_name = f"Cricket: {event_name_raw or match_type}"

            competition = event_name_raw if event_name_raw else competition_name

            add_event(
                event_name=event_name,
                event_type="cricket",
                event_category="sport",
                sport_or_category="cricket",
                date_obj=date_obj,
                source="Cricsheet",
                source_url=zip_url,
                attendance=None,
                attendance_source=None,
                home_team=team1,
                away_team=team2,
                competition=competition,
                round_name=match_type,
                notes=f"Cricsheet JSON file: {filename}; attendance not provided"
            )

            added_this_zip += 1
            total_added += 1

        print(f"{competition_name} rows added: {added_this_zip}")

    print(f"Cricsheet cricket rows added total: {total_added}")


# ============================================================
# 4. CONCERT ARCHIVES - SCG CONCERTS
# ============================================================

def scrape_concert_archives_scg(max_pages=4):
    print("\nScraping Concert Archives SCG concerts...")

    bad_lines = {
        "uploaded by",
        "embedded by",
        "setlists",
        "videos",
        "photos",
        "+ add to archive",
        "show duplicate",
        "show duplicates",
        "sydney cricket ground",
        "sydney, new south wales, australia",
        "ticket information not available.",
    }

    date_pattern = re.compile(
        r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2},\s+\d{4}$"
    )

    added = 0

    for page in range(1, max_pages + 1):
        url = CONCERT_ARCHIVES_URL if page == 1 else f"{CONCERT_ARCHIVES_URL}?page={page}"

        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                continue
        except Exception:
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        lines = [clean_text(x) for x in soup.get_text("\n").split("\n")]
        lines = [x for x in lines if x]

        current_date = None

        for line in lines:
            if date_pattern.match(line):
                current_date = line
                continue

            if not current_date:
                continue

            lower = line.lower()

            if lower in bad_lines:
                continue

            if any(lower.startswith(bad) for bad in bad_lines):
                continue

            if line in ["← Previous", "Next →"] or line.isdigit():
                continue

            date_obj = parse_date_flexible(current_date)

            if not in_year_range(date_obj):
                current_date = None
                continue

            add_event(
                event_name=line,
                event_type="concert",
                event_category="concert",
                sport_or_category="concert/performance",
                date_obj=date_obj,
                source="Concert Archives",
                source_url=url,
                attendance=None,
                attendance_source=None,
                notes="Concert Archives SCG listing"
            )

            added += 1
            current_date = None

    print(f"Concert Archives rows added: {added}")


# ============================================================
# 5. SETLIST.FM - EXTRA SCG CONCERTS
# ============================================================

def scrape_setlist_scg(max_pages=5):
    print("\nScraping setlist.fm SCG concerts...")

    added = 0

    venue_pattern = re.compile(
        r"^(.*?)\s+at\s+Sydney Cricket Ground,\s+Sydney,\s+Australia",
        re.I
    )

    date_pattern = re.compile(
        r"\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s+\d{1,2}\s+\d{4}\b"
    )

    time_pattern = re.compile(
        r"\b(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm))\b"
    )

    for page in range(1, max_pages + 1):
        url = SETLIST_BASE_URL if page == 1 else f"{SETLIST_BASE_URL}?page={page}"

        try:
            r = requests.get(url, headers=HEADERS, timeout=30)
            if r.status_code != 200:
                continue
        except Exception:
            continue

        soup = BeautifulSoup(r.text, "html.parser")
        lines = [
            clean_text(x)
            for x in soup.get_text("\n").split("\n")
            if clean_text(x)
        ]

        page_added = 0

        for i, line in enumerate(lines):
            m = venue_pattern.search(line)

            if not m:
                continue

            artist = clean_text(m.group(1))
            nearby = " ".join(lines[max(0, i - 10):i + 50])

            date_match = date_pattern.search(nearby)
            if not date_match:
                continue

            date_obj = parse_date_flexible(date_match.group(0))

            if not in_year_range(date_obj):
                continue

            time_match = time_pattern.search(nearby)
            start_time = time_match.group(1).upper().replace(" ", "") if time_match else None

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
                notes="setlist.fm SCG listing; attendance usually unavailable"
            )

            added += 1
            page_added += 1

        print(f"setlist.fm page {page}: {page_added} rows")

    print(f"setlist.fm rows added: {added}")


# ============================================================
# SAVE
# ============================================================

def save_outputs():
    df = pd.DataFrame(events)

    if df.empty:
        print("\nNo events found.")
        return

    # Clean garbage
    df = df[df["event_name"].notna()]
    df = df[df["event_name"].astype(str).str.strip() != ""]

    bad_event_names = [
        "uploaded by",
        "embedded by",
        "setlists",
        "videos",
        "photos",
    ]

    df = df[~df["event_name"].str.lower().isin(bad_event_names)]

    # Dates
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    df = df[
        (df["date"].dt.year >= START_YEAR) &
        (df["date"].dt.year <= END_YEAR)
    ]

    # Drop duplicates across overlapping sources
    df = df.drop_duplicates(
        subset=["event_name", "venue", "date", "event_type"],
        keep="first"
    )

    df["start"] = pd.to_datetime(df["start"], errors="coerce")
    df["end"] = pd.to_datetime(df["end"], errors="coerce")

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["day"] = df["date"].dt.day
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"] = df["date"].dt.dayofweek.isin([5, 6]).astype(int)

    df["start_hour"] = df["start"].dt.hour
    df["end_hour"] = df["end"].dt.hour

    df = df.sort_values(["date", "event_category", "event_type", "event_name"])
    df = df.reset_index(drop=True)

    df.insert(0, "event_id", range(1, len(df) + 1))

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

    df.to_csv(CSV_FILE, index=False, encoding="utf-8-sig", lineterminator="\n")

    conn = sqlite3.connect(DB_FILE)
    df.to_sql(TABLE_NAME, conn, if_exists="replace", index=False)
    conn.close()

    print("\nSaved SCG 2013-2019 database.")
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
    scrape_afl_tables_scg()
    scrape_ultimate_aleague_scg()
    scrape_cricsheet_scg()
    scrape_concert_archives_scg(max_pages=4)
    scrape_setlist_scg(max_pages=5)
    save_outputs()