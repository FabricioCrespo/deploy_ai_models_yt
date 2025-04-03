# load_balancer.py
from flask import Flask, request
import itertools
import requests

app = Flask(__name__)

# TODO: Add backend server URL for round-robin distribution
BACKEND_SERVERS = [
   "http://backend-flask-service:5001"
]

# Round-robin iterator for distributing requests
server_pool = itertools.cycle(BACKEND_SERVERS)

@app.route('/')
def load_balance():
    backend_url = next(server_pool)
    user_id = request.args.get("user_id", "Guest")
    response = requests.get(f"{backend_url}/", params={"user_id": user_id})
    return response.text

if __name__ == '__main__':
    # TODO: Change the port if necessary (default is 8080)
    app.run(host='0.0.0.0', port=8080)