from __future__ import print_function
import cv2
import os
import openface
import pickle
import sys
import csv

from flask import Flask, jsonify, request
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np


file_dir = os.path.dirname(os.path.realpath(__file__))
model_dir = os.path.join(file_dir, '..', 'models')
dlib_model_dir = os.path.join(model_dir, 'dlib')
openface_model_dir = os.path.join(model_dir, 'openface')
dlib_face_predictor = os.path.join(dlib_model_dir,
        "shape_predictor_68_face_landmarks.dat")
network_model = os.path.join(openface_model_dir, 'nn4.small2.v1.t7')

img_dim = 96

app = Flask(__name__)
align = openface.AlignDlib(dlib_face_predictor)
net = openface.TorchNeuralNet(network_model, imgDim=img_dim, cuda=False)


@app.route("/", methods=['POST'])
def root():
    f = request.files['face']

    #TODO: everything
    name = 'Anonymous'

    return jsonify({name: name})


def get_rep(face_image):
    bgr_img = cv2.imdecode(face_image, -1)
    assert bgr_img is not None
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgb_img)
    if bb is None:
        return None
    aligned_face = align.align(
            img_dim, rgb_img, bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    return net.forward(aligned_face)


def read_file(filename):
    with open(filename, 'r') as f:
        return f.read()


def read_fb_ids_and_names():
    with open('data/data.txt') as f:
        return [(int(row[0]), row[1])
                for row in csv.reader(f)]


def get_rep_from_file(path):
    data = read_file(path)
    np_string = np.fromstring(data, dtype='uint8')
    return get_rep(np_string)


def train():
    fb_data = read_fb_ids_and_names()
    fb_ids = [row[0] for row in fb_data]
    working_fb_ids = []
    reps = []
    for fb_id in fb_ids:
        rep = get_rep_from_file("data/{}.jpg".format(fb_id))
        if rep is not None:
            working_fb_ids.append(fb_id)
            reps.append(rep)

    le = LabelEncoder().fit(working_fb_ids)
    labels = le.transform(working_fb_ids)
    clf = SVC(kernel='linear', probability=True)
    clf.fit(reps, labels)
    fname = "classifier.pkl"
    with open(fname, 'w') as f:
        pickle.dump((le, clf), f)
    print("Classifier saved to {}".format(fname), file=sys.stderr)


def load_model():
    with open('classifier.pkl') as f:
        le, clf = pickle.load(f)
    return le, clf


def infer(rep, le, clf):
    rep = rep.reshape(1, -1)
    predictions = clf.predict_proba(rep).ravel()
    max_i = np.argmax(predictions)
    person = le.inverse_transform(max_i)
    confidence = predictions[max_i]
    return (person, confidence)


def test_infer():
    fb_data = read_fb_ids_and_names()
    fb_ids = [row[0] for row in fb_data]
    le, clf = load_model()
    num_correct = 0
    for fb_id in fb_ids:
        image_path = "data/{}.jpg".format(fb_id)
        rep = get_rep_from_file(image_path)
        if rep is not None:
            person, confidence = infer(rep, le, clf)
            print(person, confidence)
            if person == fb_id:
                num_correct += 1
    print("Infer test: {} correct out of {}".format(num_correct, len(fb_ids)))


if __name__ == '__main__':
    command = sys.argv[1]
    if command == 'run':
        app.run(host='0.0.0.0', port=8000, debug=True)
    elif command == 'train':
        train()
    elif command == 'test_infer':
        test_infer()
    else:
        print('invalid command {}'.format(sys.argv[1]))
