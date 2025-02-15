set -x

# Define checkpoint name
CKPT=Qwen/Qwen2.5-7B-Instruct
CKPT_NAME=$(echo $CKPT | rev | cut -d'/' -f1 | rev)

# Set file path (Note: Do not include .json in the file path)
FILE=./data/gemma_2b_it_annotated_processed
FILE_NAME=$(echo $FILE | rev | cut -d'/' -f1 | rev)

# Divide data based on the number of GPUs available per task
gpu_list="${CUDA_VISIBLE_DEVICES:-0}"
IFS=',' read -ra GPULIST <<< "$gpu_list"

# Define number of GPUs per task
GPUS_PER_TASK=1
CHUNKS=$((${#GPULIST[@]}/$GPUS_PER_TASK))

# Output file path for the merged results
output_file=./output/${CKPT_NAME}/${FILE_NAME}/merge.jsonl

# Check if the output file does not exist
if [ ! -f "$output_file" ]; then
    for IDX in $(seq 0 $((CHUNKS-1))); do
        # Loop through the chunks and assign GPUs for each task
        gpu_devices=$(IFS=,; echo "${GPULIST[*]:$(($IDX*$GPUS_PER_TASK)):$GPUS_PER_TASK}")
        TRANSFORMERS_OFFLINE=1 CUDA_VISIBLE_DEVICES=${gpu_devices} python3 inference_vllm.py \
            --model_path Qwen/Qwen2.5-7B-Instruct \
            --input_json ./data/gemma_2b_it_annotated_processed.json \
            --num-chunks $CHUNKS \
            --chunk-idx $IDX &
    done
    wait # Wait for all background processes to complete

    # Clear the output file if it already exists
    > "$output_file"

    # Loop through the chunks and concatenate each output file into the final output file
    for IDX in $(seq 0 $((CHUNKS-1))); do
        cat ./output/${CKPT_NAME}/${FILE_NAME}/${FILE_NAME}_${CKPT_NAME}_${CHUNKS}_${IDX}.jsonl >> "$output_file"
    done

fi
