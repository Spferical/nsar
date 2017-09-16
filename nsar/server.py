from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/", methods=['POST'])
def root():
    f = request.files['face']

    #TODO: everything
    name = 'Anonymous'

    return jsonify({name: name})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
