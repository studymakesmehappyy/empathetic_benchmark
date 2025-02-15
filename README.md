# Empathy Benchmark

This repository provides tools for running inference on the **Empathy Benchmark** dataset. The inference code is designed to evaluate consistency and empathy in dialogue responses using large language models. 

## Running Inference

### Step 1: Prepare the Input Data

The input JSON file is located under the `./data` directory. The file should be formatted according to the **Empathy Benchmark** dataset.

### Step 2: Running Inference using Bash

To run the inference, use the provided **inference_with_vllm** script. This script will process the input data and output the results with **empathy** and **consistency** scores.

### Command to run inference:
### Run the inference (for example, using GPUs 0, 1, 2, 3)
CUDA_VISIBLE_DEVICES=0,1,2,3 bash inference_with_vllm.sh $INPUT_JSON $CKPT
