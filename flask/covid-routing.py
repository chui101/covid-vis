import json

import requests
from flask import Flask
import calculate_trends



app = Flask(__name__)


@app.route("/")
@app.route("/root")
@app.route("/index")
@app.route("/home")
@app.route("/trends", methods=['GET'])
def root():
    return json.dumps(calculate_trends.calculate_trends())

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
