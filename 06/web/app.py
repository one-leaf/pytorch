from flask import Flask, render_template, jsonify, request
import random

app = Flask(__name__, static_url_path='')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/step')
def step():
    steps = request.args.get('steps', [])
    act = (random.randint(0,14),random.randint(0,14))
    info = random.random()
    result={"action":act,"info":info}      # action：下一步的动作， info：当前AI胜率
    return jsonify(result=result)

if __name__ =="__main__":
    app.run(debug=True, port=8080)