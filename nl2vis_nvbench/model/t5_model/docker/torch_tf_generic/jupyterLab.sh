#!/bin/bash

docker compose exec w266-ml jupyter lab --notebook-dir=/user --ip='*' --port=8888 --no-browser --allow-root