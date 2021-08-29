import tensorflow as tf
from cat_detector.config import CLUSTERS_TO_KEEP
from cat_detector.nn_training import prepare_clusterings_for_model
from cat_detector.paths import MODEL_NO_QUANT_TFLITE_FILE, load_clusterings, MODEL_TFLITE_FILE, UC_FILE, \
    MODEL_FILE_NO_SOFTMAX
from cat_detector.prepare_model_for_uc import prepare_model_for_uc


def save_model(model, file_path):
    with open(file_path, "wb") as f:
        f.write(model)


def convert_model_no_quant(converter):
    model = converter.convert()
    return model


def get_samples():
    clustering = [(img_name, clustering_info, clustering_ratio) for img_name, clustering_info, clustering_ratio in
                  load_clusterings() if len(clustering_info) >= CLUSTERS_TO_KEEP]
    x = prepare_clusterings_for_model(clustering, clusters_to_keep=CLUSTERS_TO_KEEP)

    def batches_generator():
        # Note: conversion to float32 is mandatory! It won't work with float64!
        for input_value in tf.data.Dataset.from_tensor_slices(x.astype("float32")).batch(1).take(100):
            yield [input_value]

    return batches_generator


def convert_model_int8_quant(converter):
    # Set the optimization flag.
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # Enforce integer only quantization
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    # Provide a representative dataset to ensure we quantize correctly.
    converter.representative_dataset = get_samples()
    model = converter.convert()

    return model


if __name__ == "__main__":
    converter = tf.lite.TFLiteConverter.from_saved_model(MODEL_FILE_NO_SOFTMAX)

    model_no_quant_tflite = convert_model_no_quant(converter)
    model_tflite = convert_model_int8_quant(converter)

    prepare_model_for_uc("model_cat_classification_clustering", model_no_quant_tflite, UC_FILE)
    prepare_model_for_uc("model_cat_classification_clustering_quant", model_tflite, UC_FILE + "_quant")

    save_model(model_no_quant_tflite, MODEL_NO_QUANT_TFLITE_FILE)
    save_model(model_tflite, MODEL_TFLITE_FILE)
