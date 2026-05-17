import pandas as pd
import random

input_file = "/Users/benmckenna/group_project/royal_randwick_events_2013_2019_clean.csv"
output_file = "/Users/benmckenna/group_project/royal_randwick_events_2013_2019_clean_with_estimated_attendance.csv"

df = pd.read_csv(input_file)

def estimate_attendance(row):
    event_name = str(row.get("event_name", "")).lower()
    event_type = str(row.get("event_type", "")).lower()
    venue = str(row.get("venue", "")).lower()

    # Biggest Royal Randwick race days
    if "everest" in event_name:
        return random.randint(60000, 70000)

    elif "queen elizabeth" in event_name:
        return random.randint(45000, 60000)

    elif "the championships" in event_name or "championships" in event_name:
        return random.randint(40000, 60000)

    elif "doncaster" in event_name:
        return random.randint(40000, 55000)

    elif "derby" in event_name:
        return random.randint(35000, 50000)

    elif "epsom" in event_name or "metropolitan" in event_name:
        return random.randint(30000, 45000)

    # General racing events
    elif event_type == "race" or "race" in event_type:
        if "stakes" in event_name or "cup" in event_name or "classic" in event_name:
            return random.randint(25000, 42000)
        else:
            return random.randint(21000, 32000)

    # Concerts at Royal Randwick
    elif event_type == "concert" or "concert" in event_type or "festival" in event_name:
        return random.randint(15000, 45000)

    # Large public/special events
    elif "festival" in event_type or "festival" in event_name:
        return random.randint(20000, 50000)

    elif "expo" in event_name or "show" in event_name:
        return random.randint(10000, 30000)

    # Default Royal Randwick estimate
    elif "randwick" in venue:
        return random.randint(21000, 35000)

    # Fallback
    else:
        return random.randint(10000, 30000)

# Update the existing estimated_attendance column
df["estimated_attendance"] = df.apply(estimate_attendance, axis=1)

# Remove duplicate attendance column if your previous script created one
if "attendance" in df.columns:
    df = df.drop(columns=["attendance"])

# Update the source column
df["attendance_source"] = (
    "Simulated estimate based on event type and typical Royal Randwick attendance ranges"
)

# Save clean CSV
df.to_csv(output_file, index=False)

print("Saved cleaned CSV as:")
print(output_file)

print("\nPreview:")
print(df[["event_name", "event_type", "estimated_attendance", "attendance_source"]].head(20))