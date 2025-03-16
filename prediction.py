import base64
import json

import numpy as np
import requests
import tensorflow as tf


# Create a TensorFlow Example
def create_tf_example(text_input):
    feature = {
        "text": tf.train.Feature(bytes_list=tf.train.BytesList(value=[text_input.encode("utf-8")]))
    }
    example = tf.train.Example(features=tf.train.Features(feature=feature))
    return example.SerializeToString()


def predict(SERVER_URL, text):
    # Convert to the Example format and encode in base64
    serialized_example = create_tf_example(text)
    encoded_example = base64.b64encode(serialized_example).decode("utf-8")

    # Format the request for TensorFlow Serving
    request_data = {"instances": [{"b64": encoded_example}]}

    # Send the prediction request
    try:
        response = requests.post(SERVER_URL, json=request_data)

        if response.status_code == 200:
            prediction = response.json()
            print("Prediction successful!")
            print(json.dumps(prediction, indent=2))
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    except requests.exceptions.ConnectionError:
        print("Connection error: Make sure TensorFlow Serving is running and accessible")

    with open("prediction_mapping.json", "r") as f:
        mapping = json.load(f)
    # Process the predictions with the mapping
    probabilities = prediction["predictions"][0]
    predicted_class_index = np.argmax(probabilities)
    confidence = probabilities[predicted_class_index]

    # Convert to the Example format and encode in base64
    serialized_example = create_tf_example(text)
    encoded_example = base64.b64encode(serialized_example).decode("utf-8")

    # Format the request for TensorFlow Serving
    request_data = {"instances": [{"b64": encoded_example}]}

    # Apply the mapping (adding 1 to match your requirement)
    # Converting to 1-indexed from 0-indexed
    adjusted_class_index = predicted_class_index + 1
    class_label = mapping.get(
        str(predicted_class_index + 1), f"Unknown Class {predicted_class_index + 1}"
    )

    print(f"Adjusted class index (1-indexed): {adjusted_class_index}")
    print(f"Predicted class: {class_label}")
    print(f"Confidence: {confidence*100:.2f}%")

    proba = [None] + probabilities
    # Display all probabilities with their labels
    print("\nProbability distribution:")
    for i in range(1, len(proba)):
        label = mapping.get(str(i), f"Unknown Class {proba[i]}")
        print(f"{label}: {proba[i]*100:.2f}%")

    return_val = class_label + " with " + str(confidence * 100) + "% confidence"
    return return_val
