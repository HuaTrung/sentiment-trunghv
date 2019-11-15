from __future__ import division, print_function
# coding=utf-8
# Flask utils
import time

from flask import Flask, request, abort, Response, render_template
from service.SentimentService import SentimentService
import json

# Define a flask app
app = Flask(__name__)
import os

print('Model loaded. Check http://127.0.0.1:5000/')

APP_ROOT = os.path.dirname(os.path.abspath(__file__))  # refers to application_top

SentimentService.load_tokenizer()
SentimentService.get_model1()


@app.route('/')
def hello_world():
    return render_template('home.html')


@app.route('/sentiment_detect', methods=['POST'])
def model_predict():
    if not request.json or not 'text' in request.json:
        abort(400)

    text = request.json['text']
    # start_time = time.time()

    seq = SentimentService.load_tokenizer().fit_on_text(text)
    # elapsed_time = time.time() - start_time
    # print('Load seq: ' + str(elapsed_time))

    model = SentimentService.get_model1()
    # start_time = time.time()

    res = model.predict(seq, batch_size=32, verbose=0)
    # elapsed_time = time.time() - start_time
    # print('Load predict: ' + str(elapsed_time))

    if res.shape[0] > 0:
        moods = {}
        elapse = res[0][0] - res[0][1]
        moods['positive'] = str(res[0][0])
        moods['negative'] = str(res[0][1])
        if elapse > 0.2:
            moods['response'] = 'positive'
        else:
            if elapse < - 0.2:
                moods['response'] = 'negative'
            else:
                moods['response'] = 'neutral'
        return Response(json.dumps(moods), status=200, mimetype='application/json')

    else:
        abort(400)


if __name__ == '__main__':
    # app.run(port=5000, debug=True)
    app.run()
    # Serve the app with gevent
    # http_server = WSGIServer(('0.0.0.0', 5000), app)
    # http_server.serve_forever()
