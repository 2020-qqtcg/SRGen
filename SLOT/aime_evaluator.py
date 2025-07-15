import re
import random
from datasets import load_dataset
from SLOT.base_evaluator import BaseEvaluator

class AIMEEvaluator(BaseEvaluator):
    def load_dataset(self, split="train", eval_samples=None):
        """Load AIME dataset"""
        eval_dataset = load_dataset("HuggingFaceH4/aime_2024", split=split)
        eval_QAs = [{'Q': x, 'A': y} 
                    for x, y in zip(eval_dataset['problem'], eval_dataset['answer'])]
        
        if eval_samples is not None and len(eval_QAs) > eval_samples:
            eval_QAs = random.sample(eval_QAs, eval_samples)
            
        return eval_QAs
    
    def reward_correct(self, item, answer):
        """Check if the answer is correct for AIME"""
        def extract_numerical_answer(text):
            """Extract numerical answer from text using multiple regex patterns."""
            # AIME answers are typically integers from 0 to 999
            
            # Define patterns to extract numerical answers with priority ordering
            patterns = [
                # Most specific patterns first
                (r"(?i:answer)\s*:\s*(\d+)", 100),  # "Answer: 123"
                (r"(?i:the answer is)\s*(\d+)", 110),  # "The answer is 123"
                (r"(?i:final answer)\s*(?:is)?\s*:?\s*(\d+)", 120),  # "Final answer: 123" or "Final answer is 123"
                (r"(?i:therefore)\s*,?\s*(?:the answer is)?\s*(\d+)", 130),  # "Therefore, 123"
                (r"(?i:so)\s*,?\s*(?:the answer is)?\s*(\d+)", 140),  # "So, 123"
                
                # Boxed answer patterns (common in math competitions)
                (r"\\boxed\{(\d+)\}", 50),  # "\boxed{123}"
                (r"\\framebox\{(\d+)\}", 60),  # "\framebox{123}"
                
                # Number at end of response
                (r"(\d+)\s*$", 250),  # "123" at very end
                (r"(\d+)\s*[\.\,]\s*$", 240),  # "123." or "123," at end
                
                # Number in parentheses or brackets
                (r"\((\d+)\)", 220),  # "(123)"
                (r"\[(\d+)\]", 230),  # "[123]"
                
                # Less specific - just numbers with some context
                (r"(?i:answer)\s*.*?(\d+)", 300),  # "answer" followed by number somewhere
                (r"(\d+)", 400),  # Just any number (lowest priority)
            ]
            
            # Try each pattern in priority order
            best_match = None
            best_priority = float('inf')
            
            for pattern, priority in patterns:
                matches = list(re.finditer(pattern, text))
                if matches:
                    # For patterns with same priority, take the last match
                    match = matches[-1]
                    # Only consider answers in valid AIME range (0-999)
                    number = int(match.group(1))
                    if 0 <= number <= 999 and priority < best_priority:
                        best_match = match
                        best_priority = priority
            
            if best_match:
                return int(best_match.group(1))
            
            return None
        
        # Extract the predicted number from the model's response
        predicted_number = extract_numerical_answer(answer)
        
        # Get the correct answer (should be a string representing an integer)
        try:
            correct_answer = int(item.get("A", "").strip())
        except ValueError:
            return -1.0
        
        # If we couldn't extract a number, return negative score
        if predicted_number is None:
            return -1.0
        
        # Compare predicted vs correct
        result_score = 1.0 if predicted_number == correct_answer else -1.0
        return result_score
    
    def get_system_prompt(self):
        """Get system prompt for AIME"""
        return """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.
The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

    def setup_logging(self, args):
        return super().setup_logging(args, benchmark_name="aime")

def main():
    # Initialize evaluator
    evaluator = AIMEEvaluator()
    args = evaluator.setup_args()
    
    # Setup environment and logging
    evaluator.setup_environment(args)
    log_file = evaluator.setup_logging(args)
    
    evaluator.load_model(args.model_path)
    
    # Set generation parameters
    generation_params = {
        "do_sample": False,
        "temperature": args.temperature if args.do_sample else None,
        "max_new_tokens": 2048  # AIME problems can require longer reasoning
    }
    
    # Run evaluation
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