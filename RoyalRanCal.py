import pdfplumber
import pandas as pd
import re
import sqlite3
import requests
import urllib3

# Fix SSL issue on your Mac/Python setup
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

pdf_url = "https://racing.australianturfclub.com.au/Content/PDF/atc-fixture-card-2025-26.pdf"
pdf_path = "atc_fixture_card_2025_26.pdf"

# Download PDF
response = requests.get(
    pdf_url,
    verify=False
)

response.raise_for_status()

with open(pdf_path, "wb") as f:
    f.write(response.content)

print("PDF downloaded.")

venues = [
    "Royal Randwick",
    "Rosehill Gardens",
    "Canterbury Park",
    "Warwick Farm",
    "Kensington"
]

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
    "DEC": "December"
}

rows = []

current_month = None
current_year = None

# Read PDF text
with pdfplumber.open(pdf_path) as pdf:
    for page in pdf.pages:
        text = page.extract_text()

        if not text:
            continue

        lines = text.split("\n")

        for line in lines:
            line = line.strip()

            # Detect month/year headings
            month_match = re.search(
                r"\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b\s+(\d{4})",
                line
            )

            if month_match:
                current_month = month_match.group(1)
                current_year = month_match.group(2)

            # Detect venue rows
            for venue in venues:
                if venue.upper() in line.upper() and current_month and current_year:

                    day_match = re.search(r"\b(\d{1,2})\b", line)

                    if day_match:
                        day = day_match.group(1)

                        rows.append({
                            "day": day,
                            "month": current_month,
                            "year": current_year,
                            "venue": venue,
                            "raw_text": line
                        })

df = pd.DataFrame(rows)

if df.empty:
    print("No fixture rows found.")
else:
    df["date"] = pd.to_datetime(
        df["day"].astype(str)
        + " "
        + df["month"].map(month_map)
        + " "
        + df["year"].astype(str),
        errors="coerce",
        dayfirst=True
    )

    df = df[["date", "venue", "raw_text"]]

    df = df.drop_duplicates()

    print(df.head(30))

    df.to_csv("atc_fixture_events.csv", index=False)

    conn = sqlite3.connect("events_database.db")

    df.to_sql(
        "atc_fixture_events",
        conn,
        if_exists="replace",
        index=False
    )

    conn.close()

    print("Saved:")
    print("- atc_fixture_events.csv")
    print("- events_database.db")