import os

import tensorflow_model_analysis as tfma
from tfx.components import (
    CsvExampleGen,
    Evaluator,
    ExampleValidator,
    Pusher,
    SchemaGen,
    StatisticsGen,
    Trainer,
    Transform,
    Tuner,
)
from tfx.dsl.components.common.resolver import Resolver
from tfx.dsl.input_resolution.strategies.latest_blessed_model_strategy import (
    LatestBlessedModelStrategy,
)
from tfx.orchestration import pipeline
from tfx.orchestration.beam.beam_dag_runner import BeamDagRunner
from tfx.orchestration.metadata import sqlite_metadata_connection_config
from tfx.proto import example_gen_pb2, pusher_pb2, trainer_pb2
from tfx.types import Channel
from tfx.types.standard_artifacts import Model, ModelBlessing

PIPELINE_NAME = "political-bias-pipeline"
SCHEMA_PIPELINE_NAME = "political-bias-tfdv-schema"

# Directory untuk menyimpan artifact yang akan dihasilkan
PIPELINE_ROOT = os.path.join("yusrilhasan-pipelines", PIPELINE_NAME)

# Path to a SQLite DB file to use as an MLMD storage.
METADATA_PATH = os.path.join("metadata", PIPELINE_NAME, "metadata.db")

# Output directory where created models from the pipeline will be exported.
SERVING_MODEL_DIR = os.path.join("yusrilhasan-serving_model_dir", PIPELINE_NAME)

DATA_ROOT = "data"

output = example_gen_pb2.Output(
    split_config=example_gen_pb2.SplitConfig(
        splits=[
            example_gen_pb2.SplitConfig.Split(name="train", hash_buckets=8),
            example_gen_pb2.SplitConfig.Split(name="eval", hash_buckets=2),
        ]
    )
)

c_input = example_gen_pb2.Input(splits=[example_gen_pb2.Input.Split(name="data", pattern="*.csv")])

eval_config = tfma.EvalConfig(
    model_specs=[tfma.ModelSpec(label_key="bias_xf")],
    slicing_specs=[tfma.SlicingSpec()],
    metrics_specs=[
        tfma.MetricsSpec(
            metrics=[
                tfma.MetricConfig(class_name="ExampleCount"),
                tfma.MetricConfig(class_name="CategoricalAccuracy"),
                tfma.MetricConfig(class_name="AUC"),
            ]
        )
    ],
)


def create_pipeline():
    example_gen = CsvExampleGen(input_base=DATA_ROOT, output_config=output, input_config=c_input)
    statistics_gen = StatisticsGen(examples=example_gen.outputs["examples"])
    schema_gen = SchemaGen(statistics=statistics_gen.outputs["statistics"])
    example_validator = ExampleValidator(
        statistics=statistics_gen.outputs["statistics"], schema=schema_gen.outputs["schema"]
    )
    transform = Transform(
        examples=example_gen.outputs["examples"],
        schema=schema_gen.outputs["schema"],
        module_file=os.path.join("political_bias_transform.py"),
    )
    trainer = Trainer(
        module_file=os.path.join("political_bias_trainer.py"),
        examples=transform.outputs["transformed_examples"],
        schema=schema_gen.outputs["schema"],
        transform_graph=transform.outputs["transform_graph"],
        train_args=trainer_pb2.TrainArgs(num_steps=100),
        eval_args=trainer_pb2.EvalArgs(num_steps=50),
    )
    model_resolver = Resolver(
        strategy_class=LatestBlessedModelStrategy,
        model=Channel(type=Model),
        model_blessing=Channel(type=ModelBlessing),
    ).with_id("latest_blessed_model_resolver")
    evaluator = Evaluator(
        examples=transform.outputs["transformed_examples"],
        model=trainer.outputs["model"],
        baseline_model=model_resolver.outputs["model"],
        eval_config=eval_config,
    )
    pusher = Pusher(
        model=trainer.outputs["model"],
        model_blessing=evaluator.outputs["blessing"],
        push_destination=pusher_pb2.PushDestination(
            filesystem=pusher_pb2.PushDestination.Filesystem(base_directory=SERVING_MODEL_DIR)
        ),
    )
    tuner = Tuner(
        module_file=os.path.join("political_bias_tuner.py"),
        examples=transform.outputs["transformed_examples"],
        transform_graph=transform.outputs["transform_graph"],
        schema=schema_gen.outputs["schema"],
        train_args=trainer_pb2.TrainArgs(splits=["train"], num_steps=500),
        eval_args=trainer_pb2.EvalArgs(splits=["eval"], num_steps=100),
    )

    return pipeline.Pipeline(
        pipeline_name=PIPELINE_NAME,
        pipeline_root=PIPELINE_ROOT,
        components=[
            example_gen,
            statistics_gen,
            schema_gen,
            example_validator,
            transform,
            trainer,
            model_resolver,
            evaluator,
            pusher,
            tuner,
        ],
        metadata_connection_config=sqlite_metadata_connection_config(METADATA_PATH),
        enable_cache=True,
        beam_pipeline_args=[],
    )


if __name__ == "__main__":
    BeamDagRunner().run(create_pipeline())
