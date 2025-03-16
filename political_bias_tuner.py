import keras_tuner as kt
import tensorflow as tf
import tensorflow_transform as tft
from tfx.components.trainer.fn_args_utils import FnArgs

# Define a function to get TunerFnResult


def get_tuner_fn_result_class():
    """Get the appropriate TunerFnResult class for the current TFX installation."""
    try:
        # Try the most common import paths
        try:
            from tfx.components.tuner.component import TunerFnResult

            return TunerFnResult
        except ImportError:
            try:
                from tfx.extensions.google_cloud_ai_platform.tuner.component import TunerFnResult

                return TunerFnResult
            except ImportError:
                try:
                    from tfx.v1.components.tuner.component import TunerFnResult

                    return TunerFnResult
                except ImportError:
                    # Define our own if none of the imports work
                    class CustomTunerFnResult:
                        def __init__(self, tuner, fit_kwargs):
                            self.tuner = tuner
                            self.fit_kwargs = fit_kwargs

                    return CustomTunerFnResult

    except Exception as e:
        print(f"Error importing TunerFnResult: {e}")

        # Define a basic version if all else fails
        class FallbackTunerFnResult:
            def __init__(self, tuner, fit_kwargs):
                self.tuner = tuner
                self.fit_kwargs = fit_kwargs

        return FallbackTunerFnResult


# Get the appropriate TunerFnResult implementation
TunerFnResult = get_tuner_fn_result_class()


LABEL_KEY = "bias"
FEATURE_KEY = "text"
NUM_CLASSES = 5
VOCAB_SIZE = 1000
embedding_dim = 16


def transformed_name(key):
    """Renaming transformed features"""
    return key + "_xf"


def gzip_reader_fn(filenames):
    """Loads compressed data"""
    return tf.data.TFRecordDataset(filenames, compression_type="GZIP")


def input_fn(file_pattern, tf_transform_output, num_epochs=1, batch_size=64):
    """Get post_transform feature & create batches of data"""

    transform_feature_spec = tf_transform_output.transformed_feature_spec().copy()

    dataset = tf.data.experimental.make_batched_features_dataset(
        file_pattern=file_pattern,
        batch_size=batch_size,
        features=transform_feature_spec,
        reader=gzip_reader_fn,
        num_epochs=num_epochs,
        label_key=transformed_name(LABEL_KEY),
    )
    return dataset


def model_builder(hp):
    """Build machine learning model with minimal tuning"""
    inputs = tf.keras.Input(shape=(1,), name=transformed_name(FEATURE_KEY), dtype=tf.string)

    # Hash the text to integer indices
    hashed_text = tf.strings.to_hash_bucket_fast(inputs, VOCAB_SIZE)

    # Reshape to ensure consistent shape
    reshaped_text = tf.reshape(hashed_text, [-1, 1])

    # Embedding layer
    word_vectors = tf.keras.layers.Embedding(input_dim=VOCAB_SIZE, output_dim=embedding_dim)(
        reshaped_text
    )

    x = tf.keras.layers.GlobalAveragePooling1D()(word_vectors)

    # Only tune one hyperparameter with explicit default value
    units = hp.Choice("units", values=[32, 64, 128], default=64)

    x = tf.keras.layers.Dense(units, activation="relu")(x)
    x = tf.keras.layers.Dense(64, activation="relu")(x)
    outputs = tf.keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # Use a default learning rate to avoid None comparison
    learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4], default=1e-2)
    model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        metrics=["accuracy"],
    )
    return model


# Critical fix for the specific error
class CustomTuner(kt.RandomSearch):
    """Custom tuner that avoids the None comparison issue"""

    def get_best_models(self, num_models=1):
        """Override to avoid None comparison"""
        if num_models is None:
            num_models = 1
        return super().get_best_models(num_models)


def tuner_fn(fn_args: FnArgs):
    """Build the tuner using the KerasTuner API with fixes for the None comparison issue."""

    # Get transform output
    tf_transform_output = tft.TFTransformOutput(fn_args.transform_graph_path)

    # Create a custom tuner that handles None values
    tuner = CustomTuner(
        model_builder,
        objective="val_accuracy",
        max_trials=3,
        directory=fn_args.working_dir,
        project_name="political_bias_tuning",
    )

    # Create training dataset
    train_dataset = input_fn(
        file_pattern=fn_args.train_files,
        tf_transform_output=tf_transform_output,
        num_epochs=1,
        batch_size=64,
    )

    # Create validation dataset
    eval_dataset = input_fn(
        file_pattern=fn_args.eval_files,
        tf_transform_output=tf_transform_output,
        num_epochs=1,
        batch_size=64,
    )

    # Explicitly set non-None values for steps
    train_steps = 100 if fn_args.train_steps is None else fn_args.train_steps
    eval_steps = 50 if fn_args.eval_steps is None else fn_args.eval_steps

    return TunerFnResult(
        tuner=tuner,
        fit_kwargs={
            "x": train_dataset,
            "validation_data": eval_dataset,
            "steps_per_epoch": train_steps,  # Use explicit non-None value
            "validation_steps": eval_steps,  # Use explicit non-None value
        },
    )
