#!/bin/bash
docker run --rm -v $(pwd)/outputs:/app/outputs stitch python3 predictive_model.py
