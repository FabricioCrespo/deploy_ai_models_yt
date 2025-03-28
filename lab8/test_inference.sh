#!/bin/bash
# Instruction on inference formart:
# https://stackoverflow.com/questions/75096564/unrecognized-content-type-parameters-format-when-serving-model-on-databricks-ex
# https://mlflow.org/docs/latest/deployment/deploy-model-locally.html#local-inference-server-spec

curl http://127.0.0.1:6002/invocations -H 'Content-Type: application/json' -d '{
  "dataframe_split": {
    "columns": ["input1", "input2", "input3", "input4"],
    "data": [[1, 2, 3, 4]]
  }
}'
