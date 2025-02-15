import json
import os
import argparse
import re
import math

from tqdm import tqdm
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

def load_json(file_path):
    """Read a JSON file and return data as a list."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def split_list(lst, n):
    """Split a list into n (roughly) equal-sized chunks."""
    chunk_size = math.ceil(len(lst) / n)  # integer division
    return [lst[i:i+chunk_size] for i in range(0, len(lst), chunk_size)]

def load_jsonl(file_path):
    """Read a JSONL file and return data as a list."""
    if not os.path.exists(file_path):
        return []
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def get_chunk(lst, n, k):
    """Get the k-th chunk from a list."""
    chunks = split_list(lst, n)
    return chunks[k]

def construct_prompt(question):
    """Construct the prompt for the large model."""
    system_prompt = "You are a model capable of evaluating consistency and empathy. Your task is to analyze the following dialogue and determine which response better aligns with both standards."
    
    task_prompt = (
        "Please evaluate both consistency and empathy. If you believe the first answer is better, output 1. "
        "If the second answer is better, output -1. If both are equally good, output 0. "
        "Only output in the exact following JSON format without any extra text: "
        "{ \"consistency_score\": integer (-1, 0, or 1), \"empathy_score\": integer (-1, 0, or 1) }"
    )
    
    return system_prompt , f"{question}\n{task_prompt}"

def generate_output_file(input_file, model_path):
    """Generate the output JSONL file path."""
    input_filename = os.path.splitext(os.path.basename(input_file))[0]   
    model_name = os.path.basename(model_path)  
    output_filename = f"{input_filename}_{model_name}_{args.num_chunks}_{args.chunk_idx}.jsonl"
    output_dir = os.path.join("./output", model_name)
    output_dir = os.path.join(output_dir, input_filename)
    os.makedirs(output_dir, exist_ok=True)
    return os.path.join(output_dir, output_filename)

def parse_model_response(response):
    """Parse the model's response to extract consistency and empathy scores."""
    match = re.search(r'\{\s*"consistency_score"\s*:\s*(-?\d+),\s*"empathy_score"\s*:\s*(-?\d+)\s*\}', response)
    if match:
        return int(match.group(1)), int(match.group(2))  
    
    # Return error message if parsing fails
    error_response = f"error: {response}"  
    return error_response, error_response 

def process_questions(input_file, model_path):
    """Process questions, get model responses, and write results to JSONL file."""
    data = load_json(input_file)
    output_file = generate_output_file(input_file, model_path)
    
    # Get the appropriate chunk of data
    data = get_chunk(data, args.num_chunks, args.chunk_idx)


    llm = LLM(model=args.model_path, tensor_parallel_size=1)
    
    # Load existing data if the output file already exists (for resuming)
    if os.path.exists(output_file):
        existing_data = load_jsonl(output_file)
        processed_questions = {entry['question'] for entry in existing_data}
    else:
        existing_data = []
        processed_questions = set()
    for entry in tqdm(data, desc="Processing questions", unit="entry"):
        question = entry['question']
        if question in processed_questions:
            print(f"Skipping already processed question: {question}")
            continue  # Skip already processed questions

        system_prompt, prompt = construct_prompt(question)

        message = [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}]
        tokenizer = AutoTokenizer.from_pretrained(args.model_path)

        text = tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )

        # Set generation parameters
        sampling_params = SamplingParams(temperature=0.7, top_p=0.95, max_tokens=4000)

        # Generate output using the model
        outputs = llm.generate([text], sampling_params)

        for k, o in enumerate(outputs):
            generated_text = o.outputs[0].text
        
        consistency_score, empathy_score = parse_model_response(generated_text)

        # Append the result
        result_entry = {
            "question": question,
            "consistency_score": consistency_score,
            "empathy_score":empathy_score
        }


        # Save data incrementally to the JSONL file
        data_json_str = json.dumps(result_entry, ensure_ascii=False)
        with open(output_file, "a", encoding="utf-8") as f:
            f.write(data_json_str + "\n")

        print(f"Processed question {question}, output: Consistency Score: {consistency_score}, Empathy Score: {empathy_score}")
    
    print(f"Processing complete, results saved to {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process questions with a given model.")
    parser.add_argument("--input_json", type=str, required=True, help="Path to the input JSON file.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model.")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    args = parser.parse_args()
    
    process_questions(args.input_json, args.model_path)
