import os
import re
import random
import argparse
import torch
from transformers import AutoTokenizer
from SLOT.modeling_qwen2_slot import Qwen2ForCausalLM

class BaseEvaluator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    def load_model(self, model_path):
        """Load model and tokenizer"""
        print(f"Loading model from: {model_path}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = Qwen2ForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation="sdpa"
        ).to("cuda")
        
    def load_dataset(self, split, eval_samples=None):
        """Abstract method to load dataset - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement load_dataset")
        
    def reward_format(self, item, answer):
        """Check if answer follows the required format"""
        pattern = r"^<think>.*?</think><answer>.*?</answer>$"
        match_obj = re.match(pattern, answer, re.DOTALL) 
        result_score = 1.25 if match_obj else -1.0
        return result_score
        
    def reward_correct(self, item, answer):
        """Abstract method to check answer correctness - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement reward_correct")
        
    def get_system_prompt(self):
        """Abstract method to get system prompt - must be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement get_system_prompt")

    def generate_with_entropy_control(self, inputs, generation_params, max_retries=5):
        """Generate text with entropy control"""
        os.environ["entropy_control"] = "True"
        os.environ["log_entropy_control"] = "True"
        
        full_completion = ""
        current_inputs = inputs.copy()
        retry_count = 0
        
        while retry_count < max_retries:
            self.model.reset_entropy_detection()
            os.environ["prompt_only"] = "True"  
            
            outputs = self.model.generate(
                **current_inputs,
                **generation_params,
            )
            
            new_tokens = outputs[0][current_inputs['input_ids'].shape[1]:]
            completion_part = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
            
            if self.model.high_entropy_detected:
                print(f"High entropy detected at retry {retry_count}, position {self.model.high_entropy_position}")
                print(f"Partial completion: {completion_part}")
                
                full_completion += completion_part
                new_text = self.tokenizer.decode(current_inputs['input_ids'][0], skip_special_tokens=True) + completion_part
                current_inputs = self.tokenizer(new_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
                
                retry_count += 1
                print(f"Continuing generation with {current_inputs['input_ids'].shape[1]} tokens")
            else:
                full_completion += completion_part
                print(f"Generation completed normally after {retry_count} retries")
                break
        
        if retry_count >= max_retries:
            print(f"Max retries ({max_retries}) reached due to high entropy, continuing with normal generation")
            
            os.environ["entropy_control"] = "False"
            os.environ["prompt_only"] = "False"
            
            self.model.reset_entropy_detection()
            
            print(f"Continuing normal generation from {current_inputs['input_ids'].shape[1]} tokens")
            final_outputs = self.model.generate(
                **current_inputs,
                **generation_params,
            )
            
            final_new_tokens = final_outputs[0][current_inputs['input_ids'].shape[1]:]
            final_completion_part = self.tokenizer.decode(final_new_tokens, skip_special_tokens=True)
            
            full_completion += final_completion_part
            print(f"Normal generation completed, added {len(final_new_tokens)} tokens")
        else:
            print(f"Generation completed normally after {retry_count} retries")
        
        os.environ["entropy_control"] = "False"
        
        return full_completion, retry_count

    def evaluate_model(self, eval_samples=None, split="test", generation_params=None, seed=42, log_file="evaluation_log.txt"):
        """Evaluate model on dataset"""
        print("Starting model evaluation...")
        self.model.eval()    
        random.seed(seed)
        
        eval_QAs = self.load_dataset(split, eval_samples)
        print(f"Evaluating {len(eval_QAs)} samples")
        
        with open(log_file, "a") as f:
            f.write(f"Number of evaluation samples: {len(eval_QAs)}\n\n")
        
        correct = 0
        format_correct = 0
        total = len(eval_QAs)
        total_retries = 0
        
        for i, qa in enumerate(eval_QAs):
            if self.model.delta is not None:
                self.model.delta = None
                
            if (i + 1) % 10 == 0:
                print(f"Evaluated {i+1}/{total} samples")
                
            prompt = qa['Q']
            prompt_text = self.tokenizer.apply_chat_template([
                {"role": "system", "content": self.get_system_prompt()},
                {"role": "user", "content": prompt}
            ], tokenize=False, add_generation_prompt=True)
            
            inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            
            use_entropy_control = os.environ.get("use_entropy_control", "False") == "True"
            if use_entropy_control:
                print(f"\n--- Sample {i+1} use_entropy_control start---")
                max_retries = int(os.environ.get("max_retries", "5"))
                completion, retry_count = self.generate_with_entropy_control(inputs, generation_params, max_retries)
                print(f"--- Sample {i+1} use_entropy_control end---")
            else:
                os.environ["prompt_only"] = "True"
                os.environ["record_prompt_entropy"] = "True"
                outputs = self.model.generate(
                    **inputs,
                    **generation_params,
                )
                completion = self.tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
                retry_count = 0
            
            format_score = self.reward_format(qa, completion)
            correct_score = self.reward_correct(qa, completion)
            
            is_format_correct = format_score > 0
            is_answer_correct = correct_score > 0
            
            if is_format_correct:
                format_correct += 1
            if is_answer_correct:
                correct += 1
            
            total_retries += retry_count
                
            with open(log_file, "a") as f:
                f.write(f"Sample {i+1}:\n")
                f.write(f"Question: {qa['Q']}\n")
                f.write(f"Model Response: {completion}\n")
                f.write(f"Correct Answer: {qa['A']}\n")
                f.write(f"Format Correct: {is_format_correct}, Answer Correct: {is_answer_correct}\n")
                f.write(f"Retry Count: {retry_count}\n\n")
                
            print(f"\n--- Sample {i+1} ---")
            print("Question:", qa['Q'])
            print("Model Response:", completion)
            print("Correct Answer:", qa['A'])
            print(f"Format Correct: {is_format_correct}, Answer Correct: {is_answer_correct}")
            print(f"Retry Count: {retry_count}")
        
        accuracy = correct / total if total > 0 else 0
        format_accuracy = format_correct / total if total > 0 else 0
        avg_retries = total_retries / total if total > 0 else 0
        
        print(f"\nEvaluation Results (Samples: {total}):")
        print(f"Answer Accuracy: {accuracy:.4f}")
        print(f"Format Accuracy: {format_accuracy:.4f}")
        print(f"Total Retries: {total_retries}")
        print(f"Average Retries per Sample: {avg_retries:.2f}")
        
        with open(log_file, "a") as f:
            f.write(f"Evaluation Results (Samples: {total}):\n")
            f.write(f"Answer Accuracy: {accuracy:.4f}\n")
            f.write(f"Format Accuracy: {format_accuracy:.4f}\n")
            f.write(f"Total Retries: {total_retries}\n")
            f.write(f"Average Retries per Sample: {avg_retries:.2f}\n")
        
        return accuracy, format_accuracy

    @staticmethod
    def setup_args():
        """Setup command line arguments"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, default="/ssdwork/huyang/r1/simple_GRPO_debug/slot_gsm8k/models/Qwen2.5-7B", help="Path to the model")
        parser.add_argument("--eval_samples", type=int, default=None, help="Number of samples to evaluate, None for full evaluation")
        parser.add_argument("--split", type=str, default="test", choices=["test", "train"], help="Dataset split to evaluate on")
        parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling for generation")
        parser.add_argument("--temperature", type=float, default=0.9, help="Generation temperature")
        parser.add_argument("--seed", type=int, default=42, help="Random seed for consistent evaluation samples")
        parser.add_argument("--use_entropy_control", action="store_true", help="Enable entropy-based early stopping and continuation")
        parser.add_argument("--entropy_threshold", type=float, default=5.0, help="Entropy threshold for early stopping")
        parser.add_argument("--max_retries", type=int, default=5, help="Maximum number of retries for entropy-controlled generation")
        parser.add_argument("--times", type=int, default=0, help="Number of optimization iterations")
        parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for optimization")
        parser.add_argument("--record_entropy", action="store_true", help="Whether to record entropy analysis")
        parser.add_argument("--entropy_output_file", type=str, default="my_analysis.jsonl", help="Output file for entropy analysis")
        parser.add_argument("--entropy_weight", type=float, default=0.1, help="Weight for entropy loss")
        return parser.parse_args()

    @staticmethod
    def setup_environment(args):
        """Setup environment variables"""
        os.environ["times"] = str(args.times)
        os.environ["lr"] = str(args.lr)
        os.environ["record_entropy"] = str(args.record_entropy).lower()
        os.environ["entropy_output_file"] = args.entropy_output_file
        os.environ["tokenizer_path"] = args.model_path
        os.environ["entropy_threshold"] = str(args.entropy_threshold)
        os.environ["entropy_weight"] = str(args.entropy_weight)
        
        if args.use_entropy_control:
            os.environ["use_entropy_control"] = "True"
            os.environ["entropy_threshold"] = str(args.entropy_threshold)
            os.environ["max_retries"] = str(args.max_retries)
            print(f"Entropy control enabled with threshold: {args.entropy_threshold}, max retries: {args.max_retries}")
        else:
            os.environ["use_entropy_control"] = "False"

    @staticmethod
    def setup_logging(args, benchmark_name: str = "base"):
        """Setup logging directory and file"""
        log_dir = "logs"
        os.makedirs(log_dir, exist_ok=True)
        entropy_suffix = f"_entropy_{args.entropy_threshold}_weight_{args.entropy_weight}" if args.use_entropy_control else ""
        log_file = os.path.join(log_dir, f"log_{benchmark_name}_times_{args.times}_lr_{args.lr}{entropy_suffix}.txt")
        
        with open(log_file, "w") as f:
            f.write(f"Model Path: {args.model_path}\n")
            f.write(f"Times: {args.times}\n")
            f.write(f"LR: {args.lr}\n")
            f.write(f"Record Entropy: {args.record_entropy}\n")
            f.write(f"Entropy Output File: {args.entropy_output_file}\n")
            f.write(f"Entropy Weight: {args.entropy_weight}\n")
            f.write(f"Eval Samples: {'All' if args.eval_samples is None else args.eval_samples}\n")
            f.write(f"Dataset Split: {args.split}\n")
            f.write(f"Do Sample: {args.do_sample}\n")
            f.write(f"Temperature: {args.temperature}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write(f"Use Entropy Control: {args.use_entropy_control}\n")
            f.write(f"Entropy Threshold: {args.entropy_threshold}\n")
            f.write(f"Max Retries: {args.max_retries}\n\n")
        
        return log_file