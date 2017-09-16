import cv2
import os
import openface
import pickle

from flask import Flask, jsonify, request
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder


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
    bgr_img = cv2.imdecode(face_image)
    assert bgr_img is not None
    rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
    bb = align.getLargestFaceBoundingBox(rgb_img)
    assert bb is not None
    aligned_face = align.align(
            img_dim, rgb_img, bb,
            landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
    return net.forward(aligned_face)


def train():
    # TODO: read images, names, IDs, etc
    raise NotImplementedError()
    labels = None
    le = LabelEncoder().fit(labels)
    num_labels = le.transform(labels)
    clf = SVC(C=1, kernel='linear', probability=True)
    clf.fit(embeddings, num_labels)
    fname = "classifier.pkl"
    with open("classifier.pkl") as f:
        pickle.dump((le, clf), f)


def load_model():
    with open('classifier.pkl') as f:
        le, clf = pickle.load(f)
    return le, clf


def infer(img, le, clf):
    r = get_rep(img)
    bbx, rep = (r[0], r[1].reshape(1, -1))
    predictions = clf.predict_proba(rep).ravel()
    max_i = np.argmax(predictions)
    person = le.inverse_transform(max_i)
    confidence = predictions[max_i]
    return (person, confidence)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
