import os

from transformers import AutoTokenizer

import torch
import re
from datasets import load_dataset
import random
import argparse

from modeling_qwen2_slot import Qwen2ForCausalLM

def reward_correct(item, answer):
    """
    Extract letter choice (A, B, C, D) from the LLM response and compare with correct answer.
    Uses multiple regex patterns with priority ordering, similar to the reference code.
    """
    def extract_letter_answer(text):
        """Extract letter choice from text using multiple regex patterns."""
        # Define the letter choices pattern
        letter_choices = r"(?P<letter>[ABCD])"
        
        # Create regex patterns with priorities (lower number = higher priority)
        patterns = [
            # Most specific patterns first
            (rf"(?i:answer)\s*:\s*{letter_choices}", 100),  # "Answer: A"
            (rf"(?i:the answer is)\s*{letter_choices}", 110),  # "The answer is A"
            (rf"(?i:final answer)\s*(?:is)?\s*:?\s*{letter_choices}", 120),  # "Final answer: A" or "Final answer is A"
            (rf"(?i:therefore)\s*,?\s*(?:the answer is)?\s*{letter_choices}", 130),  # "Therefore, A" or "Therefore, the answer is A"
            (rf"(?i:so)\s*,?\s*(?:the answer is)?\s*{letter_choices}", 140),  # "So, A" or "So, the answer is A"
            
            # Letter at start of line patterns
            (rf"^\s*{letter_choices}\s*[\.\)\,\:]", 200),  # "A." or "A)" or "A," or "A:" at start
            (rf"\n\s*{letter_choices}\s*[\.\)\,\:]", 210),  # Same but after newline
            
            # Letter with parentheses
            (rf"\({letter_choices}\)", 220),  # "(A)"
            
            # Letter at end of response
            (rf"{letter_choices}\s*$", 250),  # "A" at very end
            (rf"{letter_choices}\s*[\.\,]\s*$", 240),  # "A." or "A," at end
            
            # Less specific - just letter with some context
            (rf"(?i:answer)\s*.*?{letter_choices}", 300),  # "answer" followed by letter somewhere
            (rf"{letter_choices}", 400),  # Just the letter anywhere (lowest priority)
        ]
        
        # Try each pattern in priority order
        best_match = None
        best_priority = float('inf')
        
        for pattern, priority in patterns:
            matches = list(re.finditer(pattern, text))
            if matches:
                # For patterns with same priority, take the last match
                match = matches[-1]
                if priority < best_priority:
                    best_match = match
                    best_priority = priority
        
        if best_match:
            return best_match.group('letter').upper()
        
        return None
    
    # Extract the predicted letter from the model's response
    predicted_letter = extract_letter_answer(answer)
    
    # Get the correct answer (should be a single letter like "A", "B", "C", "D")
    correct_answer = item.get("A", "").strip().upper()
    
    # If we couldn't extract a letter, return negative score
    if predicted_letter is None:
        return -1.0
    
    # Compare predicted vs correct
    result_score = 1.0 if predicted_letter == correct_answer else -1.0
    return result_score

def reward_format(item, answer):
    pattern = r"^<think>.*?</think><answer>.*?</answer>$"
    match_obj = re.match(pattern, answer, re.DOTALL) 
    result_score = 1.25 if match_obj else -1.0
    return result_score

def evaluate_model(model, tokenizer, eval_samples=None, split="train", generation_params=None, seed=42, log_file="evaluation_log.txt"):
    """Evaluates the model's performance on the GPQA dataset."""
    print("Starting model evaluation...")
    model.eval()    
    random.seed(seed)
    
    # Load the evaluation dataset
    eval_dataset = load_dataset("Idavidrein/gpqa", subset="gpqa_diamond", split=split)
    filterd_data = []
    for row in eval_dataset:
        choices = [
            row['Incorrect Answer 1'],
            row['Incorrect Answer 2'],
            row['Incorrect Answer 3'],
            row['Correct Answer']
        ]
        choices = [choice.strip() for choice in choices]

        random.shuffle(choices)
        choices_dict = dict(
            A=choices[0], B=choices[1], C=choices[2], D=choices[3], Question=row["Question"]
        )
        correct_answer_idx = choices.index(row['Correct Answer'].strip())

        task_template = """
        {Question}

        A) {A}
        B) {B}
        C) {C}
        D) {D}
        """

        task = task_template.format(
            Question=choices_dict['Question'],
            A=choices_dict['A'],
            B=choices_dict['B'],
            C=choices_dict['C'],
            D=choices_dict['D'])

        answer = "A" if correct_answer_idx == 0 \
            else "B" if correct_answer_idx == 1 \
            else "C" if correct_answer_idx == 2 \
            else "D"

        filterd_data.append({
            "question": task,
            "answer": answer,
            "golden_answer": row["Explanation"]
        })
    eval_QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} 
                for x,y in zip(filterd_data['question'], filterd_data['answer'])]
    
    # Randomly select samples for evaluation if specified
    if eval_samples is not None and len(eval_QAs) > eval_samples:
        eval_QAs = random.sample(eval_QAs, eval_samples)
    
    # Print the actual number of samples being evaluated
    print(f"Evaluating {len(eval_QAs)} samples")
    
    # Append evaluation info to the log
    with open(log_file, "a") as f:
        f.write(f"Number of evaluation samples: {len(eval_QAs)}\n\n")
    
    system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.
"""
    
    correct = 0
    format_correct = 0
    total = len(eval_QAs)
    
    for i, qa in enumerate(eval_QAs):
        if (i + 1) % 10 == 0:
            print(f"Evaluated {i+1}/{total} samples")
            
        prompt = qa['Q']
        prompt_text = tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ], tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(model.device)
        # target_ids = tokenizer(qa['A'], return_tensors="pt", add_special_tokens=False).to(model.device)
        
        os.environ["prompt_only"] = "True" # Ensure this env var is handled correctly if needed elsewhere
        outputs = model.generate(
            **inputs,
            **generation_params,
        )
        
        completion = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        
        # Check format and correctness
        format_score = reward_format(qa, completion)
        correct_score = reward_correct(qa, completion)
        
        is_format_correct = format_score > 0
        is_answer_correct = correct_score > 0
        
        if is_format_correct:
            format_correct += 1
        if is_answer_correct:
            correct += 1
            
        # Log sample information
        with open(log_file, "a") as f:
            f.write(f"Sample {i+1}:\n")
            f.write(f"Question: {qa['Q']}\n")
            f.write(f"Model Response: {completion}\n")
            f.write(f"Correct Answer: {qa['A']}\n")
            f.write(f"Format Correct: {is_format_correct}, Answer Correct: {is_answer_correct}\n\n")
            
        # Print detailed information for every sample
        print(f"\n--- Sample {i+1} ---")
        print("Question:", qa['Q'])
        print("Model Response:", completion)
        print("Correct Answer:", qa['A'])
        print(f"Format Correct: {is_format_correct}, Answer Correct: {is_answer_correct}")
    
    accuracy = correct / total if total > 0 else 0
    format_accuracy = format_correct / total if total > 0 else 0
    
    print(f"\nEvaluation Results (Samples: {total}):")
    print(f"Answer Accuracy: {accuracy:.4f}")
    print(f"Format Accuracy: {format_accuracy:.4f}")
    
    # Log overall results
    with open(log_file, "a") as f:
        f.write(f"Evaluation Results (Samples: {total}):\n")
        f.write(f"Answer Accuracy: {accuracy:.4f}\n")
        f.write(f"Format Accuracy: {format_accuracy:.4f}\n")
    
    return accuracy, format_accuracy

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, default="/ssdwork/huyang/r1/simple_GRPO_debug/slot_gsm8k/models/Qwen2.5-7B", help="Path to the model")
    parser.add_argument("--eval_samples", type=int, default=None, help="Number of samples to evaluate, None for full evaluation")
    parser.add_argument("--split", type=str, default="train", choices=["test", "train"], help="Dataset split to evaluate on")
    parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling for generation")
    parser.add_argument("--temperature", type=float, default=0.9, help="Generation temperature")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for consistent evaluation samples")
    args = parser.parse_args()
    # args.eval_samples = 30
    
    print(f"Loading model from: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    
    # Ensure same model loading parameters as training if applicable
    model = Qwen2ForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        _attn_implementation="sdpa" # Use 'flash_attention_2' if available and preferred
    ).to("cuda") # Consider adding device management if multiple GPUs
    
    # Set generation parameters
    generation_params = {
        "do_sample": False,
        "temperature": args.temperature if args.do_sample else None,
        "max_new_tokens": 512 # Added a sensible default, adjust if needed
    }

    # Get environment variables (consider passing as args instead for clarity)
    times = os.environ.get("times", "3")
    lr = os.environ.get("lr", "0.001")
    
    # Create log directory and file
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"log_analysis_times_{times}_lr_{lr}.txt")
    
    # Log basic information
    with open(log_file, "w") as f: # Use 'w' to overwrite for a new run
        f.write(f"Model Path: {args.model_path}\n")
        f.write(f"Times (from env): {times}\n")
        f.write(f"LR (from env): {lr}\n")
        f.write(f"Eval Samples: {'All' if args.eval_samples is None else args.eval_samples}\n")
        f.write(f"Dataset Split: {args.split}\n")
        f.write(f"Do Sample: {args.do_sample}\n")
        f.write(f"Temperature: {args.temperature}\n")
        f.write(f"Seed: {args.seed}\n\n")
    
    # Call evaluate_model, passing the log file path
    accuracy, format_accuracy = evaluate_model(
        model, 
        tokenizer, 
        eval_samples=args.eval_samples, 
        split=args.split, 
        generation_params=generation_params, 
        seed=args.seed,
        log_file=log_file # Pass log file path
    )
    
    # Log final results (already done inside evaluate_model, but can add a summary here if needed)
    print(f"Evaluation complete. Results logged to {log_file}")

if __name__ == "__main__":
    main()
