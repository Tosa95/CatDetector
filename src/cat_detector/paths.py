import os
import pickle
import json

DATA_FOLDER = "C:/Users/david/Documents/cats"
IMAGES_FOLDER = os.path.join(DATA_FOLDER, "CroquettesBlu")
CLUSTERING_FILE = os.path.join(DATA_FOLDER, "clustered.pickle")
LABELS_FILE = os.path.join(DATA_FOLDER, "labels.json")
MODEL_FILE = os.path.join(DATA_FOLDER, "model")
MODEL_FILE_NO_SOFTMAX = os.path.join(DATA_FOLDER, "model_no_softmax")
MODEL_NO_QUANT_TFLITE_FILE = os.path.join(DATA_FOLDER, "model_tflite")
MODEL_TFLITE_FILE = os.path.join(DATA_FOLDER, "model_tflite_quant")

NN_UC_FOLDER = "C:\\Users\\david\\OneDrive\\ProgettiCorrenti\\ESPCam\\src\\nn"
UC_FILE = os.path.join(NN_UC_FOLDER, "model_cat_classification_clustering")

def load_clusterings():
    with open(CLUSTERING_FILE, "rb") as f:
        return pickle.load(f)

def load_labels():
    with open(LABELS_FILE, "rt") as f:
        return json.load(f)

def load_uc_model_data():
    with open(MODEL_TFLITE_FILE, "rb") as f:
        return f.read()