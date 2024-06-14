from flask import Flask, request
from src.api.routes import api_blueprint

app = Flask(__name__)
app.register_blueprint(api_blueprint)

@app.before_request
def before_request():
    print(f"Received request: {request.method} {request.url}")

if __name__ == '__main__':
    app.run(debug=True)
