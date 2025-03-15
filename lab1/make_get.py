import requests

# API endpoint
url = "http://localhost:3000/api/v1/analysis/"

# JSON payload with the actual image path
payload = {
    "uri": "hello.png"
}

# Headers to specify JSON content
headers = {
    "Content-Type": "application/json"
}

# Sending POST request with JSON data
response = requests.get(url, json=payload, headers=headers)

# Print the response
if response.status_code == 200:
    print("Response:", response.json())  # Assuming the API returns JSON
else:
    print("Error:", response.status_code, response.text)
