from flask import Flask, request
from flask_cors import CORS
from src.api.routes import api_blueprint
import uvicorn

app = Flask(__name__)

# Specify the origins allowed
origins = [
    "http://localhost:8080",
    "localhost:8080",
]

CORS(app, origins=origins, supports_credentials=True, allow_methods=["*"], allow_headers=["*"])


app.register_blueprint(api_blueprint)

@app.before_request
def before_request():
    print(f"Received request: {request.method} {request.url}")

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000, log_level="info") # Change with your host
