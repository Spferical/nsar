from __future__ import print_function
import sys
from flask import Flask, jsonify, request


app = Flask(__name__)


@app.route("/", methods=['POST'])
def root():
    f = request.files['face']

    #TODO: everything
    name = 'Anonymous'

    return jsonify({name: name})


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'run':
        app.run(host='0.0.0.0', port=8000, debug=True)
    else:
        print('invalid command {}'.format(sys.argv[1]))
