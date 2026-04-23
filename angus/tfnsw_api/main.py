import requests
import yaml

# Load API key
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

API_KEY = config["tfnsw"]["api_key"]

# TfNSW Traffic Volume API endpoint
url = "https://api.transport.nsw.gov.au/v1/traffic_volume"

# Simple test query (station list)
params = {
    "q": "SELECT * FROM road_traffic_counts_station_reference LIMIT 5",
    "format": "json"
}

headers = {
    "Authorization": f"apikey {API_KEY}"
}

response = requests.get(url, headers=headers, params=params)

print("Status code:", response.status_code)
print(response.json())