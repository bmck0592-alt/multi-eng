import matplotlib.pyplot as plt

dates = ["2026-05-20", "2026-05-21", "2026-05-22", "2026-05-23"]
historic_average = [12000, 13500, 12800, 14000]
predicted_traffic = [15000, 13800, 17000, 13000]

print("\nTraffic Forecast Results")
print("Date\t\tHistoric\tPredicted\tDifference\tAssistance")

for date, historic, predicted in zip(dates, historic_average, predicted_traffic):
    difference = predicted - historic
    assistance_needed = predicted > historic * 1.25

    assistance_text = "Yes" if assistance_needed else "No"

    print(f"{date}\t{historic}\t\t{predicted}\t\t{difference}\t\t{assistance_text}")

print("\nAssistance Check")
for date, historic, predicted in zip(dates, historic_average, predicted_traffic):
    if predicted > historic * 1.25:
        print(
            f"{date}: Assistance needed — predicted traffic is {predicted}, "
            f"compared to historic average {historic}."
        )
    else:
        print(f"{date}: No assistance likely needed.")

plt.figure(figsize=(10, 6))

plt.plot(dates, historic_average, marker="o", label="Historic Average")
plt.plot(dates, predicted_traffic, marker="o", label="Predicted Traffic")

plt.xlabel("Date")
plt.ylabel("Traffic Count")
plt.title("Traffic Forecast: Predicted vs Historic")
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig("traffic_forecast.png", dpi=300)
plt.show()