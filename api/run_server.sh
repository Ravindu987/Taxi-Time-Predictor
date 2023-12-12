#! /bin/bash

# Run the server
cd api
BENTOML_CONFIG="./bentoml_configuration.yaml" bentoml serve create_service.py