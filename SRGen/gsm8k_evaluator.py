import re
import random
from datasets import load_dataset
from SRGen.base_evaluator import BaseEvaluator
from transformers import AutoTokenizer

class GSM8KEvaluator(BaseEvaluator):
    def load_dataset(self, split="test", eval_samples=None, *args, **kwargs):
        """Load GSM8K dataset"""
        eval_dataset = load_dataset("openai/gsm8k", "main", split=split)
        eval_QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} 
                    for x,y in zip(eval_dataset['question'], eval_dataset['answer'])]
        
        if eval_samples is not None and len(eval_QAs) > eval_samples:
            eval_QAs = random.sample(eval_QAs, eval_samples)

        return eval_QAs
    
    def reward_correct(self, item, answer):
        """Check if the answer is correct for GSM8K"""
        from math_verify import parse, verify, ExprExtractionConfig
        pattern = r'\d+\.\d+|\d+/\d+|\d+'
        nums = re.findall(pattern, answer)
        if len(nums) == 0:
            return -1.0
        lastnum = nums[-1]
        
        ans_parsed = None
        ground_truth_parsed = None

        try:
            ans_parsed = parse(lastnum, extraction_config=[ExprExtractionConfig()])
        except Exception as e:
            return -1.0

        try:
            ground_truth_parsed = parse(item["A"], extraction_config=[ExprExtractionConfig()])
        except Exception as e:
            return -1.0
        
        if ans_parsed is None or ground_truth_parsed is None:
            return -1.0

        verification_result = verify(ans_parsed, ground_truth_parsed)
        result_score = 1.0 if verification_result else -1.0
        return result_score

    # def reward_correct(self, item, answer):
    #     """
    #     Checks if the answer for a GSM8K item is correct.
    #     This function extracts a numerical value from the model's response and 
    #     compares it to the ground truth.
    #     """
    #     def extract_numerical_answer(text: str):
    #         """
    #         Extracts a numerical answer from text using multiple regex patterns.
    #         Optimized for GSM8K format (handles commas, decimals, etc.).
    #         """
    #         # GSM8K answers can contain commas and/or decimal points.
    #         # Define a more robust number pattern to handle them.
    #         number_pattern = r"[\d,]+(?:\.\d+)?"

    #         # Define regex patterns with priorities.
    #         # A lower priority number means a higher logical priority.
    #         patterns = [
    #             # 1. Highest priority: Escape the literal braces with double braces.
    #             (r"\\boxed{{({number_pattern})}}", 10),
    #             (r"\\framebox{{({number_pattern})}}", 20),

    #             # 2. These patterns are fine as they don't contain literal braces.
    #             (r"(?i:final answer|the answer is)\s*:?\s*({number_pattern})", 30),
    #             (r"({number_pattern})\s*$", 40),
    #             (r"({number_pattern})", 100)
    #         ]

    #         best_match_val = None
    #         best_priority = float('inf')

    #         for pattern_template, priority in patterns:
    #             # Embed the number_pattern into the regex template.
    #             pattern = pattern_template.format(number_pattern=number_pattern)
                
    #             # Find all matches for the current pattern.
    #             matches = list(re.finditer(pattern, text))
                
    #             if matches:
    #                 # If matches are found, prioritize the last one as it's often the conclusion.
    #                 last_match_str = matches[-1].group(1)
                    
    #                 # Update if a higher-priority match is found.
    #                 if priority < best_priority:
    #                     best_priority = priority
    #                     # Clean the extracted number string (remove commas) and convert to float.
    #                     try:
    #                         cleaned_val = float(last_match_str.replace(",", ""))
    #                         best_match_val = cleaned_val
    #                     except ValueError:
    #                         # If conversion fails, this match is invalid, so skip it.
    #                         continue

    #         return best_match_val

    #     # Extract the predicted number from the model's response.
    #     predicted_number = extract_numerical_answer(answer)

    #     # Get and clean the ground truth answer from the item.
    #     try:
    #         # The ground truth answer also needs to be cleaned (commas removed) and
    #         # converted to float for consistent comparison.
    #         correct_answer_str = item.get("A", "").strip()
    #         correct_answer = float(correct_answer_str.replace(",", ""))
    #     except (ValueError, TypeError):
    #         # If the ground truth answer is malformed, we can't score.
    #         return 0.0 # Or return a specific error code/None.

    #     # If no number could be extracted from the model's output, give a negative score.
    #     if predicted_number is None:
    #         return -1.0

    #     # Compare the predicted and correct numbers.
    #     # Use float comparison with a small tolerance for precision issues.
    #     if abs(predicted_number - correct_answer) < 1e-9:
    #         return 1.0
    #     else:
    #         return -1.0
    
    def get_system_prompt(self):
        """Get system prompt for GSM8K"""
        return """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""
        # return """You are a helpful assistant. Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Answer: $NUMBER' (without quotes) where NUMBER is the final number answer. Think step by step before answering."""

    def setup_logging(self, args):
        return super().setup_logging(args, benchmark_name="gsm8k")

def main():
    
    # Initialize evaluator
    evaluator = GSM8KEvaluator()
    args = evaluator.setup_args()
    
    # Setup environment and logging
    evaluator.setup_environment(args)
    log_file = evaluator.setup_logging(args)
    if not args.parallel:
        evaluator.load_model(args.model_path, device=args.device)
    else:
        evaluator.model_path = args.model_path

    evaluator.tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    masked_token_ids = [
        evaluator.tokenizer.encode(token, add_special_tokens=False)[0] 
        for token in ["system", "user", "assistant", ":", "\n"]
    ] if args.mask_special_tokens else None
    
    # Set generation parameters
    generation_params = {
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "max_new_tokens": args.max_new_tokens,
        "masked_token_ids": masked_token_ids
    }
    
    # Run evaluation multiple times and take average if specified
    if args.average > 1:
        print(f"Running evaluation {args.average} times and taking average...")
        accuracies = []
        format_accuracies = []
        
        for run in range(args.average):
            print(f"Run {run + 1}/{args.average}...")
            
            if args.parallel:
                accuracy, format_accuracy = evaluator.evaluate_model_parallel(
                    eval_samples=args.eval_samples,
                    split=args.split,
                    generation_params=generation_params,
                    seed=args.seed,
                    log_file=log_file,
                    max_parallel_gpus=args.max_parallel_gpus
                )
            else:
                accuracy, format_accuracy = evaluator.evaluate_model(
                    eval_samples=args.eval_samples,
                    split=args.split,
                    generation_params=generation_params,
                    seed=args.seed,
                    log_file=log_file
                )
            
            accuracies.append(accuracy)
            format_accuracies.append(format_accuracy)
            print(f"Run {run + 1} - Accuracy: {accuracy:.4f}, Format Accuracy: {format_accuracy:.4f}")
        
        # Calculate averages
        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_format_accuracy = sum(format_accuracies) / len(format_accuracies)
        
        # Log average results
        with open(log_file, "a") as f:
            f.write(f"\n=== AVERAGE RESULTS ({args.average} runs) ===\n")
            f.write(f"Individual Accuracies: {[f'{acc:.4f}' for acc in accuracies]}\n")
            f.write(f"Individual Format Accuracies: {[f'{acc:.4f}' for acc in format_accuracies]}\n")
            f.write(f"Average Accuracy: {avg_accuracy:.4f}\n")
            f.write(f"Average Format Accuracy: {avg_format_accuracy:.4f}\n")
            f.write(f"Accuracy Std Dev: {(sum((x - avg_accuracy)**2 for x in accuracies) / len(accuracies))**0.5:.4f}\n")
            f.write(f"Format Accuracy Std Dev: {(sum((x - avg_format_accuracy)**2 for x in format_accuracies) / len(format_accuracies))**0.5:.4f}\n")
        
        print(f"Average Accuracy: {avg_accuracy:.4f} (±{(sum((x - avg_accuracy)**2 for x in accuracies) / len(accuracies))**0.5:.4f})")
        print(f"Average Format Accuracy: {avg_format_accuracy:.4f} (±{(sum((x - avg_format_accuracy)**2 for x in format_accuracies) / len(format_accuracies))**0.5:.4f})")
        
    else:
        # Run evaluation once (original behavior)
        if args.parallel:
            print("Running parallel evaluation across multiple GPUs...")
            accuracy, format_accuracy = evaluator.evaluate_model_parallel(
                eval_samples=args.eval_samples,
                split=args.split,
                generation_params=generation_params,
                seed=args.seed,
                log_file=log_file,
                max_parallel_gpus=args.max_parallel_gpus
            )
        else:
            print("Running sequential evaluation...")
            accuracy, format_accuracy = evaluator.evaluate_model(
                eval_samples=args.eval_samples,
                split=args.split,
                generation_params=generation_params,
                seed=args.seed,
                log_file=log_file
            )
    
    print(f"Evaluation complete. Results logged to {log_file}")

if __name__ == "__main__":
    main()