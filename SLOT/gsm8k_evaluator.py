import re
import random
from datasets import load_dataset
from SLOT.base_evaluator import BaseEvaluator

class GSM8KEvaluator(BaseEvaluator):
    def load_dataset(self, split="test", eval_samples=None):
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
    
    def get_system_prompt(self):
        """Get system prompt for GSM8K"""
        return """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

    def setup_logging(self, args):
        return super().setup_logging(args, benchmark_name="gsm8k")

def main():
    
    # Initialize evaluator
    evaluator = GSM8KEvaluator()
    args = evaluator.setup_args()
    
    # Setup environment and logging
    evaluator.setup_environment(args)
    log_file = evaluator.setup_logging(args)
    evaluator.load_model(args.model_path)
    
    # Set generation parameters
    generation_params = {
        "do_sample": False,
        "temperature": args.temperature if args.do_sample else None,
        "max_new_tokens": 512
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