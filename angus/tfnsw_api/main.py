import requests
import yaml
import json

with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

API_KEY = config["tfnsw"]["api_key"]

url = "https://api.transport.nsw.gov.au/v1/traffic_volume"

query = """
SELECT DISTINCT year
FROM road_traffic_counts_yearly_summary
WHERE station_key = 55432
ORDER BY year DESC
"""

params = {
    "q": query,
    "format": "json"
}

headers = {
    "Authorization": f"apikey {API_KEY}"
}

response = requests.get(url, headers=headers, params=params)

print("Status code:", response.status_code)
print("Request URL:", response.url)
print("Raw response:")
print(response.text)

if response.status_code == 200:
    print(json.dumps(response.json(), indent=2))
else:
    print("Request failed.")
