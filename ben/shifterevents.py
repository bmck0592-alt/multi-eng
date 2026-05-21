import pandas as pd
from pathlib import Path

# File paths
input_file = Path("/Users/benmckenna/group_project/ben/events_cleaned.csv")
output_file = Path("/Users/benmckenna/group_project/ben/events_cleaned_AEDT.csv")

print("Reading from:", input_file)

# Check the file exists
if not input_file.exists():
    raise FileNotFoundError(f"Could not find file: {input_file}")

# Load CSV
df = pd.read_csv(input_file)

# Convert UTC times into Sydney local time
# This automatically handles:
# AEST = UTC+10
# AEDT = UTC+11
df["start"] = pd.to_datetime(df["start"], utc=True).dt.tz_convert("Australia/Sydney")
df["end"] = pd.to_datetime(df["end"], utc=True).dt.tz_convert("Australia/Sydney")

# Remove timezone info so the CSV looks cleaner
df["start"] = df["start"].dt.tz_localize(None)
df["end"] = df["end"].dt.tz_localize(None)

# Recalculate time-based columns using Sydney local time
df["hour"] = df["start"].dt.hour
df["day_of_week"] = df["start"].dt.dayofweek
df["month"] = df["start"].dt.month
df["year"] = df["start"].dt.year
df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

# Recalculate duration in hours
df["duration_hours"] = (df["end"] - df["start"]).dt.total_seconds() / 3600

# Save new CSV
df.to_csv(output_file, index=False)

print("Done!")
print("Saved corrected file to:", output_file)