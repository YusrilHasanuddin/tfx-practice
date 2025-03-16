FROM tensorflow/serving:latest

COPY ./yusrilhasan-serving_model_dir /models
ENV MODEL_NAME=political-bias-detection-model

ENV MODEL_BASE_PATH=/models
COPY ./config /model_config
# Copy config files
COPY ./models.config /models/config/models.config
COPY ./model_warmup.json /models/warmup/model_warmup.json

ENV MONITORING_CONFIG="/model_config/prometheus.config"
ENV PORT=8500
RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server --port=8501 --rest_api_port=${PORT} \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
--monitoring_config_file=${MONITORING_CONFIG} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh

# Start TensorFlow Serving with model warmup
CMD tensorflow_model_server \
  --port=8500 \
  --rest_api_port=${PORT} \
  --model_config_file=/models/config/models.config \
  --monitoring_config_file=/models/config/prometheus.config \
  --enable_batching=true

EXPOSE 8500 8501
