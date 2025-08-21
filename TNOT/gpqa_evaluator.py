import re
import random
from datasets import load_dataset
from TNOT.base_evaluator import BaseEvaluator

class GPQAEvaluator(BaseEvaluator):
    def load_dataset(self, split="train", eval_samples=None, **kwargs):
        """Load GPQA dataset"""
        eval_dataset = load_dataset("Idavidrein/gpqa", "gpqa_diamond", split=split)
        filterd_data = {
            "question": [],
            "answer": [],
            "golden_answer": []
        }
        
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

            task_template = """{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()

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

            filterd_data['question'].append(task)
            filterd_data['answer'].append(answer)
            filterd_data['golden_answer'].append(row["Explanation"])

        eval_QAs = [{'Q':x, 'A':y} 
                    for x,y in zip(filterd_data['question'], filterd_data['answer'])]
        
        if eval_samples is not None and len(eval_QAs) > eval_samples:
            eval_QAs = random.sample(eval_QAs, eval_samples)
            
        return eval_QAs
    
    def reward_correct(self, item, answer):
        """Check if the answer is correct for GPQA"""
        def extract_letter_answer(text):
            """Extract letter choice from text using multiple regex patterns."""
            letter_choices = r"(?P<letter>[ABCD])"
            
            patterns = [
                (rf"(?i:answer)\s*:\s*{letter_choices}", 100),
                (rf"(?i:the answer is)\s*{letter_choices}", 110),
                (rf"(?i:final answer)\s*(?:is)?\s*:?\s*{letter_choices}", 120),
                (rf"(?i:therefore)\s*,?\s*(?:the answer is)?\s*{letter_choices}", 130),
                (rf"(?i:so)\s*,?\s*(?:the answer is)?\s*{letter_choices}", 140),
                (rf"^\s*{letter_choices}\s*[\.\)\,\:]", 200),
                (rf"\n\s*{letter_choices}\s*[\.\)\,\:]", 210),
                (rf"\({letter_choices}\)", 220),
                (rf"{letter_choices}\s*$", 250),
                (rf"{letter_choices}\s*[\.\,]\s*$", 240),
                (rf"(?i:answer)\s*.*?{letter_choices}", 300),
                (rf"{letter_choices}", 400),
            ]
            
            best_match = None
            best_priority = float('inf')
            
            for pattern, priority in patterns:
                matches = list(re.finditer(pattern, text))
                if matches:
                    match = matches[-1]
                    if priority < best_priority:
                        best_match = match
                        best_priority = priority
            
            if best_match:
                return best_match.group('letter').upper()
            
            return None
        
        predicted_letter = extract_letter_answer(answer)
        correct_answer = item.get("A", "").strip().upper()
        
        if predicted_letter is None:
            return -1.0
        
        result_score = 1.0 if predicted_letter == correct_answer else -1.0
        return result_score
    
    def get_system_prompt(self):
        """Get system prompt for GPQA"""
        return """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering."""

    def setup_logging(self, args):
        return super().setup_logging(args, benchmark_name="gpqa")

def main():
    # Initialize evaluator
    evaluator = GPQAEvaluator()
    args = evaluator.setup_args()
    
    # Setup environment and logging
    evaluator.setup_environment(args)
    log_file = evaluator.setup_logging(args)
    
    evaluator.load_model(args.model_path, device=args.device)

    masked_token_ids = [
        evaluator.tokenizer.encode(token, add_special_tokens=False)[0] 
        for token in ["system", "user", "assistant", ":", "\n"]
    ] if args.mask_special_tokens else None
    
    # Set generation parameters
    generation_params = {
        "do_sample": args.do_sample,
        "temperature": args.temperature if args.do_sample else None,
        "top_p": 0.95 if args.do_sample else None,
        "max_new_tokens": args.max_new_tokens,
        "masked_token_ids": masked_token_ids
    }

    print("Generation parameters:", generation_params)
    
    # Run evaluation (parallel or sequential)
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