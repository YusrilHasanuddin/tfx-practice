import tensorflow as tf
LABEL_KEY = "bias"
FEATURE_KEY = "text"
def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"
def preprocessing_fn(inputs):
    """
    Preprocess input features into transformed features
    
    Args:
        inputs: map from feature keys to raw features.
    
    Return:
        outputs: map from feature keys to transformed features.    
    """
    
    outputs = {}
    
    outputs[transformed_name(FEATURE_KEY)] = tf.strings.lower(inputs[FEATURE_KEY])
    
    # Ensure the label tensor has the correct shape
    labels = tf.one_hot(inputs[LABEL_KEY], depth=5)
    outputs[transformed_name(LABEL_KEY)] = tf.squeeze(labels, axis=-2)
    # outputs[transformed_name(LABEL_KEY)] = tf.one_hot(inputs[LABEL_KEY], depth=5)
    
    return outputs
