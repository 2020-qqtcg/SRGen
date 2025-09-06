import re
import random
from typing import Optional, List, Tuple
from datasets import load_dataset
from TNOT.base_evaluator import BaseEvaluator
from transformers import AutoTokenizer

class AIMEEvaluator(BaseEvaluator):
    def load_dataset(self, split="train", eval_samples=None, version="2024"):
        """Load AIME dataset"""
        if version == "2024":
            eval_dataset = load_dataset("HuggingFaceH4/aime_2024", split=split)
        elif version == "2025":
            eval_dataset = load_dataset("yentinglin/aime_2025", split=split)
        else:
            raise ValueError(f"Invalid version: {version}")
        
        eval_QAs = [{'Q': x, 'A': y} 
                    for x, y in zip(eval_dataset['problem'], eval_dataset['answer'])]
        
        if eval_samples is not None and len(eval_QAs) > eval_samples:
            eval_QAs = random.sample(eval_QAs, eval_samples)
            
        return eval_QAs
    
    def _extract_boxed_answer(self, text: str) -> Optional[int]:
        """Extract answer from boxed LaTeX expressions (highest priority)"""
        # Based on lighteval's LaTeX extraction patterns
        boxed_patterns = [
            r"\\boxed\{(\d+)\}",                    # \boxed{123}
            r"\$\\boxed\{(\d+)\}\$",                # $\boxed{123}$
            r"\$\$\\boxed\{(\d+)\}\$\$",            # $$\boxed{123}$$
            r"\\\[\\boxed\{(\d+)\}\\\]",            # \[\boxed{123}\]
            r"\\framebox\{(\d+)\}",                 # \framebox{123}
        ]
        
        for pattern in boxed_patterns:
            matches = re.findall(pattern, text)
            if matches:
                # Take the last match (most likely the final answer)
                number = int(matches[-1])
                if 0 <= number <= 999:
                    return number
        return None
    
    def _extract_final_answer_patterns(self, text: str) -> Optional[int]:
        """Extract from 'final answer is' patterns (medium-high priority)"""
        # Based on lighteval's final answer extraction patterns
        patterns = [
            r"(?i:final\s+answer\s+is)\s*:?\s*(\d+)(?:\s*\.?\s*I\s+hope)?",  # "final answer is: 123. I hope"
            r"(?i:the\s+final\s+answer\s+is)\s*:?\s*(\d+)(?:\s*\.?\s*I\s+hope)?",
            r"(?i:my\s+final\s+answer\s+is)\s*:?\s*(\d+)",
            r"(?i:therefore,?\s+the\s+final\s+answer\s+is)\s*:?\s*(\d+)",
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                number = int(matches[-1])
                if 0 <= number <= 999:
                    return number
        return None
    
    def _extract_answer_patterns(self, text: str) -> Optional[int]:
        """Extract from general answer patterns (medium priority)"""
        patterns = [
            r"(?i:answer)\s*:?\s*(\d+)",                    # "Answer: 123"
            r"(?i:answer)\s*:.*?(\d+)",                     # "Answer: The value is 123" (more flexible)
            r"(?i:the\s+answer\s+is)\s*:?\s*(\d+)",         # "The answer is 123"
            r"(?i:the\s+answer\s+is).*?(\d+)",              # "The answer is value 123" (more flexible)
            r"(?i:answer)\s*=\s*(\d+)",                     # "Answer = 123"
            r"\*\*(?i:answer)\*\*\s*:?\s*(\d+)",           # "**Answer**: 123"
            r"\*\*(?i:answer)\*\*\s*:.*?(\d+)",            # "**Answer**: The value is 123" (more flexible)
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                number = int(matches[-1])
                if 0 <= number <= 999:
                    return number
        return None
    
    def _extract_conclusion_patterns(self, text: str) -> Optional[int]:
        """Extract from conclusion patterns (medium priority)"""
        patterns = [
            r"(?i:therefore)\s*,?\s*(\d+)",                 # "Therefore, 123"
            r"(?i:thus)\s*,?\s*(\d+)",                      # "Thus, 123"
            r"(?i:hence)\s*,?\s*(\d+)",                     # "Hence, 123"
            r"(?i:so)\s*,?\s*(\d+)",                        # "So, 123"
            r"(?i:we\s+get)\s*:?\s*(\d+)",                 # "We get 123"
            r"(?i:we\s+have)\s*:?\s*(\d+)",                # "We have 123"
            r"(?i:we\s+find)\s*(?:that)?\s*:?\s*(\d+)",    # "We find that 123"
            r"(?i:this\s+gives)\s*(?:us)?\s*:?\s*(\d+)",   # "This gives us 123"
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            if matches:
                number = int(matches[-1])
                if 0 <= number <= 999:
                    return number
        return None
    
    def _extract_equals_patterns(self, text: str) -> Optional[int]:
        """Extract from equals patterns (lower priority)"""
        # Look for = followed by a number near the end
        pattern = r"=\s*(\d+)(?:\s*[\.!]?)?\s*(?:\n|$)"
        matches = re.findall(pattern, text, re.MULTILINE)
        if matches:
            number = int(matches[-1])
            if 0 <= number <= 999:
                return number
        return None
    
    def _extract_end_patterns(self, text: str) -> Optional[int]:
        """Extract numbers at the end of text (lowest priority)"""
        # Number at the very end, possibly with punctuation
        patterns = [
            r"(\d+)\s*[\.!]?\s*$",                         # "123." or "123!" at end
            r"(\d+)\s*$",                                   # "123" at very end
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, text.strip())
            if matches:
                number = int(matches[-1])
                if 0 <= number <= 999:
                    return number
        return None
    
    def extract_numerical_answer(self, text: str) -> Optional[int]:
        """
        Extract numerical answer using hierarchical pattern matching.
        Based on lighteval's extraction strategy with priority ordering.
        """
        if not text or not text.strip():
            return None
        
        # Clean the text
        text = text.strip()
        
        # Try extraction methods in priority order (highest to lowest)
        extraction_methods = [
            self._extract_boxed_answer,           # Priority 1: Boxed answers
            self._extract_final_answer_patterns,  # Priority 2: "Final answer is" patterns  
            self._extract_answer_patterns,        # Priority 3: General answer patterns
            self._extract_conclusion_patterns,    # Priority 4: Conclusion patterns
            self._extract_equals_patterns,        # Priority 5: Equals patterns
            self._extract_end_patterns,           # Priority 6: End patterns
        ]
        
        for method in extraction_methods:
            try:
                result = method(text)
                if result is not None:
                    return result
            except (ValueError, IndexError, AttributeError):
                continue
        
        return None
    
    def reward_correct(self, item, answer):
        """Check if the answer is correct for AIME"""
        # Extract the predicted number from the model's response
        predicted_number = self.extract_numerical_answer(answer)
        
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
    
    def reward_format(self, item, answer):
        """Check if answer follows the required format for AIME"""
        # Check if we can extract a valid numerical answer
        extracted_answer = self.extract_numerical_answer(answer)
        
        if extracted_answer is not None:
            # Give higher score for boxed answers (preferred format)
            if self._extract_boxed_answer(answer) is not None:
                return 1.5  # Bonus for using boxed format
            elif self._extract_final_answer_patterns(answer) is not None:
                return 1.25  # Good format with "final answer is"
            elif self._extract_answer_patterns(answer) is not None:
                return 1.0   # Standard answer format
            else:
                return 0.5   # Answer found but not in ideal format
        
        return -1.0  # No valid answer found

    def get_system_prompt(self):
        """Get system prompt for AIME (based on lighteval's prompt)"""
        return """You are a helpful assistant. Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{ANSWER}$. I hope it is correct' (without quotes) where ANSWER is just the final number that solves the problem. Think step by step before answering."""

    def setup_logging(self, args):
        if args.version == "2024":
            return super().setup_logging(args, benchmark_name="aime_2024")
        elif args.version == "2025":
            return super().setup_logging(args, benchmark_name="aime_2025")
        else:
            raise ValueError(f"Invalid version: {args.version}")

def main():
    # Initialize evaluator
    evaluator = AIMEEvaluator()
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
                    version=args.version,
                    max_parallel_gpus=args.max_parallel_gpus
                )
            else:
                accuracy, format_accuracy = evaluator.evaluate_model(
                    eval_samples=args.eval_samples,
                    split=args.split,
                    generation_params=generation_params,
                    seed=args.seed,
                    log_file=log_file,
                    version=args.version
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
                version=args.version,
                max_parallel_gpus=args.max_parallel_gpus
            )
        else:
            print("Running sequential evaluation...")
            accuracy, format_accuracy = evaluator.evaluate_model(
                eval_samples=args.eval_samples,
                split=args.split,
                generation_params=generation_params,
                seed=args.seed,
                log_file=log_file,
                version=args.version
            )
    
    print(f"Evaluation complete. Results logged to {log_file}")

if __name__ == "__main__":
    main()