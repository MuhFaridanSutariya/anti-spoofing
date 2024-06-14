from flask import Flask, request
from src.api.routes import api_blueprint
import uvicorn


app = Flask(__name__)
app.register_blueprint(api_blueprint)

@app.before_request
def before_request():
    print(f"Received request: {request.method} {request.url}")

if __name__ == '__main__':
    uvicorn.run("app:app", host="0.0.0.0", port=5000, log_level="info") # Change with your host
