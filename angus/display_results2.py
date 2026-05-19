import matplotlib.pyplot as plt

# Example data — later replace with Random Forest output
dates = ["2026-05-20", "2026-05-21", "2026-05-22", "2026-05-23"]
historic_average = [12000, 13500, 12800, 14000]
predicted_traffic = [15000, 13800, 17000, 13000]

# Calculate values
differences = []
assistance_needed = []

for historic, predicted in zip(historic_average, predicted_traffic):
    difference = predicted - historic
    differences.append(difference)

    if predicted > historic * 1.25:
        assistance_needed.append("YES")
    else:
        assistance_needed.append("NO")

# Print terminal version
print("\nTraffic Forecast Results")
print("Date\t\tHistoric\tPredicted\tDifference\tAssistance Needed")

for i in range(len(dates)):
    print(
        f"{dates[i]}\t"
        f"{historic_average[i]}\t\t"
        f"{predicted_traffic[i]}\t\t"
        f"{differences[i]}\t\t"
        f"{assistance_needed[i]}"
    )

# Create display window
fig, ax = plt.subplots(figsize=(11, 6))
ax.axis("off")

fig.suptitle(
    "Traffic Forecast Assistance Display",
    fontsize=18,
    fontweight="bold"
)

station_text = "Station 55432 — Cleveland Street, West of Anzac Parade"
ax.text(
    0.5,
    0.92,
    station_text,
    ha="center",
    va="center",
    fontsize=12
)

# Summary numbers
latest_index = 0

summary_text = (
    f"Selected Date: {dates[latest_index]}\n\n"
    f"Historic Average: {historic_average[latest_index]:,}\n"
    f"Predicted Traffic: {predicted_traffic[latest_index]:,}\n"
    f"Difference: {differences[latest_index]:+,}\n"
    f"Assistance Needed: {assistance_needed[latest_index]}"
)

ax.text(
    0.5,
    0.68,
    summary_text,
    ha="center",
    va="center",
    fontsize=16,
    bbox=dict(boxstyle="round,pad=0.8", edgecolor="black", facecolor="white")
)

# Table data
table_data = []

for i in range(len(dates)):
    table_data.append([
        dates[i],
        f"{historic_average[i]:,}",
        f"{predicted_traffic[i]:,}",
        f"{differences[i]:+,}",
        assistance_needed[i]
    ])

column_labels = [
    "Date",
    "Historic Avg",
    "Predicted",
    "Difference",
    "Assistance?"
]

table = ax.table(
    cellText=table_data,
    colLabels=column_labels,
    loc="lower center",
    cellLoc="center",
    colLoc="center",
    bbox=[0.05, 0.05, 0.9, 0.38]
)

table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.5)

plt.tight_layout()
plt.savefig("traffic_forecast_display.png", dpi=300)
plt.show()