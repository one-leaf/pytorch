from flask import Flask, render_template, jsonify, request

app = Flask(__name__, static_url_path='')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/step')
def step():
    actions = request.args.get('actions', [])
    result={"action":(1,1),"info":0.8}      # action：下一步的动作， info：当前AI胜率
    return jsonify(result=result)

if __name__ =="__main__":
    app.run(debug=True, port=8080)