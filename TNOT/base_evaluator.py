import os
import re
import random
import argparse
import torch
import torch.multiprocessing as mp
import json
import time
from typing import List, Dict, Any, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from TNOT.tnot_decorator import enable_tnot

class BaseEvaluator:
    def __init__(self):
        self.model = None
        self.tokenizer = None
        
    @staticmethod
    def detect_available_gpus() -> List[int]:
        """Detect available GPUs and return their indices"""
        if not torch.cuda.is_available():
            print("CUDA is not available, using CPU")
            return []
        
        gpu_count = torch.cuda.device_count()
        available_gpus = []
        
        for i in range(gpu_count):
            try:
                # Test if GPU is accessible
                torch.cuda.set_device(i)
                torch.cuda.empty_cache()
                available_gpus.append(i)
                print(f"GPU {i}: {torch.cuda.get_device_name(i)} - Available")
            except Exception as e:
                print(f"GPU {i}: Not available - {str(e)}")
                
        print(f"Found {len(available_gpus)} available GPUs: {available_gpus}")
        return available_gpus
    
    @staticmethod
    def partition_data(data: List[Dict], num_partitions: int) -> List[List[Dict]]:
        """Partition data into roughly equal chunks for parallel processing"""
        if num_partitions <= 1:
            # Even for single partition, we need to add global_idx
            data_with_idx = []
            for i, item in enumerate(data):
                item_with_idx = item.copy()
                item_with_idx['global_idx'] = i
                data_with_idx.append(item_with_idx)
            return [data_with_idx]
            
        chunk_size = len(data) // num_partitions
        remainder = len(data) % num_partitions
        
        partitions = []
        start_idx = 0
        
        for i in range(num_partitions):
            # Distribute remainder among first few partitions
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end_idx = start_idx + current_chunk_size
            
            # Add global indices to track original positions
            partition_data = []
            for j, item in enumerate(data[start_idx:end_idx]):
                item_with_idx = item.copy()
                item_with_idx['global_idx'] = start_idx + j
                partition_data.append(item_with_idx)
                
            partitions.append(partition_data)
            start_idx = end_idx
            
        # Print partition info
        for i, partition in enumerate(partitions):
            print(f"Partition {i}: {len(partition)} samples (indices {partition[0]['global_idx']}-{partition[-1]['global_idx']})")
            
        return partitions
        
    def load_model(self, model_path, device="cuda:0"):
        """Load model and tokenizer with automatic model type detection"""
        print(f"Loading model from: {model_path}")
        self.model_path = model_path  # Store for parallel processes
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        # Auto-detect model type based on config
        config = AutoConfig.from_pretrained(model_path)
        model_type = config.model_type.lower()
        
        print(f"Detected model type: {model_type}")
        
        # Create TNOT-enabled model class
        TNOTModelClass = enable_tnot(AutoModelForCausalLM)
        
        print(f"Loading model with universal TNOT implementation (model_type: {model_type})...")
        self.model = TNOTModelClass.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            _attn_implementation="flash_attention_2",
            device_map=device,
            trust_remote_code=True
        )
            
        # Check if this is a Phi model and log special handling
        if self.is_phi_model():
            print("Phi model detected: Will use combined system+user prompt format")
        
    def is_phi_model(self):
        """Check if the current model is a Phi model"""
        if self.model is None:
            return False
        
        # Check model type from config
        model_type = getattr(self.model.config, 'model_type', '').lower()
        if model_type in ['phi', 'phi3']:
            return True
            
        # Check model class name
        class_name = self.model.__class__.__name__.lower()
        if 'phi' in class_name:
            return True
            
        return False
    
    def build_prompt_text(self, prompt):
        """Build prompt text"""
        # Standard format for other models
        prompt_text = self.tokenizer.apply_chat_template([
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": prompt}
        ], tokenize=False, add_generation_prompt=True)
        return prompt_text
        
    def load_dataset(self, split, eval_samples=None, **kwargs):
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
            self.model.prompt_only = True
            
            outputs = self.model.generate(
                **current_inputs,
                **generation_params,
            )
            
            new_tokens = outputs[0][current_inputs['input_ids'].shape[1]:]
            completion_part = self.tokenizer.decode(new_tokens, skip_special_tokens=True)

            del outputs
            torch.cuda.empty_cache()
            
            if self.model.high_entropy_detected:
                print(f"High entropy detected at retry {retry_count}, position {self.model.high_entropy_position}")
                print(f"Partial completion: {completion_part}")
                
                # Add the partial completion to full_completion
                full_completion += completion_part
                
                old_inputs = current_inputs
                new_text = self.tokenizer.decode(current_inputs['input_ids'][0], skip_special_tokens=True) + completion_part
                current_inputs = self.tokenizer(new_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
                del old_inputs  # 释放旧的inputs
                
                retry_count += 1
                print(f"Continuing generation with {current_inputs['input_ids'].shape[1]} tokens")
            else:
                full_completion += completion_part
                print(f"Generation completed normally after {retry_count} retries")
                break
        
        if retry_count >= max_retries:
            print(f"Max retries ({max_retries}) reached due to high entropy, continuing with normal generation")
            
            os.environ["entropy_control"] = "False"
            self.model.prompt_only = False
            
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

    def evaluate_model(self, eval_samples=None, split="test", generation_params=None, seed=42, log_file="evaluation_log.txt", version=None):
        """Evaluate model on dataset"""
        print("Starting model evaluation...")
        self.model.eval()    
        random.seed(seed)
        
        eval_QAs = self.load_dataset(split, eval_samples, version=version)
        print(f"Evaluating {len(eval_QAs)} samples")
        
        with open(log_file, "a") as f:
            f.write(f"Number of evaluation samples: {len(eval_QAs)}\n\n")
            f.write(f"Start time: {time.time()}\n")
        
        correct = 0
        format_correct = 0
        total = len(eval_QAs)
        total_retries = 0
        
        for i, qa in enumerate(eval_QAs):
            self.model.reset_entropy_detection()
            self.model.reset_model_parameters()
                
            if (i + 1) % 10 == 0:
                print(f"Evaluated {i+1}/{total} samples")
                
            prompt = qa['Q']
            prompt_text = self.build_prompt_text(prompt)
            
            inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
            
            use_entropy_control = os.environ.get("use_entropy_control", "False") == "True"
            if use_entropy_control:
                print(f"\n--- Sample {i+1} use_entropy_control start---")
                max_retries = int(os.environ.get("max_retries", "5"))
                completion, retry_count = self.generate_with_entropy_control(inputs, generation_params, max_retries)
                print(f"--- Sample {i+1} use_entropy_control end---")
            else:
                self.model.prompt_only = True
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
            f.write(f"End time: {time.time()}\n")
            f.write(f"Evaluation Results (Samples: {total}):\n")
            f.write(f"Answer Accuracy: {accuracy:.4f}\n")
            f.write(f"Format Accuracy: {format_accuracy:.4f}\n")
            f.write(f"Total Retries: {total_retries}\n")
            f.write(f"Average Retries per Sample: {avg_retries:.2f}\n")
        
        return accuracy, format_accuracy
    
    def evaluate_partition(self, gpu_id: int, partition_data: List[Dict], generation_params: Dict, 
                          seed: int, log_file: str, temp_results_file: str) -> Dict[str, Any]:
        """Evaluate a partition of data on a specific GPU"""
        try:
            # Set device for this process
            device = f"cuda:{gpu_id}"
            torch.cuda.set_device(gpu_id)
            
            print(f"Process {gpu_id}: Starting evaluation on {device} with {len(partition_data)} samples")
            
            # Load model on this GPU
            self.load_model(os.environ.get("model_path"), device)
            self.model.eval()
            random.seed(seed)
            
            correct = 0
            format_correct = 0
            total = len(partition_data)
            total_retries = 0
            results = []
            
            for i, qa in enumerate(partition_data):
                self.model.reset_entropy_detection()
                self.model.reset_model_parameters()
                
                global_idx = qa['global_idx']
                
                if (i + 1) % 5 == 0:
                    print(f"GPU {gpu_id}: Evaluated {i+1}/{total} samples")
                    
                prompt = qa['Q']
                prompt_text = self.build_prompt_text(prompt)
                
                inputs = self.tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).to(self.model.device)
                
                use_entropy_control = os.environ.get("use_entropy_control", "False") == "True"
                if use_entropy_control:
                    max_retries = int(os.environ.get("max_retries", "5"))
                    completion, retry_count = self.generate_with_entropy_control(inputs, generation_params, max_retries)
                else:
                    self.model.prompt_only = True
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
                
                # Store result for later sorting and logging
                result = {
                    'global_idx': global_idx,
                    'gpu_id': gpu_id,
                    'question': qa['Q'],
                    'model_response': completion,
                    'correct_answer': qa['A'],
                    'is_format_correct': is_format_correct,
                    'is_answer_correct': is_answer_correct,
                    'retry_count': retry_count
                }
                results.append(result)
                
                print(f"GPU {gpu_id} - Sample {global_idx+1}: Format={is_format_correct}, Answer={is_answer_correct}, Retries={retry_count}")
            
            # Save temporary results
            temp_result = {
                'gpu_id': gpu_id,
                'correct': correct,
                'format_correct': format_correct,
                'total': total,
                'total_retries': total_retries,
                'results': results
            }
            
            with open(temp_results_file, 'w') as f:
                json.dump(temp_result, f)
                
            print(f"GPU {gpu_id}: Completed evaluation. Accuracy: {correct/total:.4f}, Format: {format_correct/total:.4f}")
            return temp_result
            
        except Exception as e:
            print(f"Error in GPU {gpu_id}: {str(e)}")
            import traceback
            traceback.print_exc()
            return {
                'gpu_id': gpu_id,
                'correct': 0,
                'format_correct': 0,
                'total': 0,
                'total_retries': 0,
                'results': [],
                'error': str(e)
            }
    
    def evaluate_model_parallel(self, eval_samples=None, split="test", generation_params=None, 
                               seed=42, log_file="evaluation_log.txt", version=None, 
                               max_parallel_gpus=None) -> Tuple[float, float]:
        """Parallel evaluation across multiple GPUs"""
        print("Starting parallel model evaluation...")
        random.seed(seed)
        # Detect available GPUs
        available_gpus = self.detect_available_gpus()
        
        if not available_gpus:
            print("No GPUs available, falling back to single-threaded CPU evaluation")
            return self.evaluate_model(eval_samples, split, generation_params, seed, log_file, version)
        
        # Determine number of parallel processes
        if max_parallel_gpus is not None:
            num_processes = min(max_parallel_gpus, len(available_gpus))
            available_gpus = available_gpus[:num_processes]
        else:
            num_processes = len(available_gpus)
            
        print(f"Using {num_processes} GPUs for parallel evaluation: {available_gpus}")
        
        # Load dataset
        eval_QAs = self.load_dataset(split, eval_samples, version=version)
        print(f"Evaluating {len(eval_QAs)} samples across {num_processes} GPUs")
        
        # Partition data
        partitions = self.partition_data(eval_QAs, num_processes)
        
        # Set up temporary files for results
        temp_dir = os.environ.get("TEMP_PARALLEL_FILE", "temp_parallel_results")
        os.makedirs(temp_dir, exist_ok=True)
        temp_files = [os.path.join(temp_dir, f"gpu_{gpu_id}_results.json") for gpu_id in available_gpus]
        
        # Store model path for subprocesses
        if hasattr(self, 'model_path'):
            os.environ["model_path"] = self.model_path
        else:
            print("Warning: model_path not found, make sure to call load_model first")
        
        # Start parallel processes
        print("Starting parallel evaluation processes...")
        processes = []
        
        # Use multiprocessing to run evaluation on each GPU
        mp.set_start_method('spawn', force=True)
        
        for i, gpu_id in enumerate(available_gpus):
            args = (gpu_id, partitions[i], generation_params, seed, log_file, temp_files[i])
            p = mp.Process(target=self._run_evaluation_process, args=args)
            p.start()
            processes.append(p)
            
        # Wait for all processes to complete
        for p in processes:
            p.join()
            
        # Collect and merge results
        print("Collecting results from all GPUs...")
        return self._collect_and_merge_results(temp_files, log_file, temp_dir)
    
    def _run_evaluation_process(self, gpu_id: int, partition_data: List[Dict], 
                               generation_params: Dict, seed: int, log_file: str, temp_results_file: str):
        """Wrapper method to run evaluation in a separate process"""
        try:
            # Create a new evaluator instance for this process
            evaluator = self.__class__()
            result = evaluator.evaluate_partition(gpu_id, partition_data, generation_params, 
                                                 seed, log_file, temp_results_file)
        except Exception as e:
            print(f"Process error on GPU {gpu_id}: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _collect_and_merge_results(self, temp_files: List[str], log_file: str, temp_dir: str) -> Tuple[float, float]:
        """Collect results from all GPUs and write ordered logs"""
        all_results = []
        total_correct = 0
        total_format_correct = 0
        total_samples = 0
        total_retries = 0
        
        # Load all temporary results
        for temp_file in temp_files:
            if os.path.exists(temp_file):
                try:
                    with open(temp_file, 'r') as f:
                        result = json.load(f)
                        
                    if 'error' in result:
                        print(f"GPU {result['gpu_id']} had error: {result['error']}")
                        continue
                        
                    total_correct += result['correct']
                    total_format_correct += result['format_correct']
                    total_samples += result['total']
                    total_retries += result['total_retries']
                    
                    all_results.extend(result['results'])
                    
                except Exception as e:
                    print(f"Error loading result file {temp_file}: {str(e)}")
            else:
                print(f"Warning: Result file {temp_file} not found")
        
        # Sort results by global index to maintain order
        all_results.sort(key=lambda x: x['global_idx'])
        
        # Write ordered log file
        with open(log_file, "a") as f:
            f.write(f"Number of evaluation samples: {total_samples}\n")
            f.write(f"Parallel evaluation across {len(temp_files)} GPUs\n\n")
            
            for result in all_results:
                f.write(f"Sample {result['global_idx']+1}:\n")
                f.write(f"GPU: {result['gpu_id']}\n")
                f.write(f"Question: {result['question']}\n")
                f.write(f"Model Response: {result['model_response']}\n")
                f.write(f"Correct Answer: {result['correct_answer']}\n")
                f.write(f"Format Correct: {result['is_format_correct']}, Answer Correct: {result['is_answer_correct']}\n")
                f.write(f"Retry Count: {result['retry_count']}\n\n")
        
        # Calculate final metrics
        accuracy = total_correct / total_samples if total_samples > 0 else 0
        format_accuracy = total_format_correct / total_samples if total_samples > 0 else 0
        avg_retries = total_retries / total_samples if total_samples > 0 else 0
        
        print(f"\nParallel Evaluation Results (Samples: {total_samples}):")
        print(f"Answer Accuracy: {accuracy:.4f}")
        print(f"Format Accuracy: {format_accuracy:.4f}")
        print(f"Total Retries: {total_retries}")
        print(f"Average Retries per Sample: {avg_retries:.2f}")
        
        # Write summary to log
        with open(log_file, "a") as f:
            f.write(f"Parallel Evaluation Results (Samples: {total_samples}):\n")
            f.write(f"Answer Accuracy: {accuracy:.4f}\n")
            f.write(f"Format Accuracy: {format_accuracy:.4f}\n")
            f.write(f"Total Retries: {total_retries}\n")
            f.write(f"Average Retries per Sample: {avg_retries:.2f}\n")
        
        # Clean up temporary files
        try:
            import shutil
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Could not clean up temporary directory: {str(e)}")
        
        return accuracy, format_accuracy

    @staticmethod
    def setup_args():
        """Setup command line arguments"""
        parser = argparse.ArgumentParser()
        parser.add_argument("--model_path", type=str, default="/ssdwork/huyang/r1/simple_GRPO_debug/slot_gsm8k/models/Qwen2.5-7B", help="Path to the model")
        parser.add_argument("--device", type=str, default="cuda:0", help="Device to run the model on (e.g., cuda:0, cpu)")
        parser.add_argument("--eval_samples", type=int, default=None, help="Number of samples to evaluate, None for full evaluation")
        parser.add_argument("--split", type=str, default="test", choices=["test", "train"], help="Dataset split to evaluate on")
        parser.add_argument("--do_sample", action="store_true", help="Whether to use sampling for generation")
        parser.add_argument("--temperature", type=float, default=0.9, help="Generation temperature")
        parser.add_argument("--max_new_tokens", type=int, default=4096, help="Maximum number of new tokens to generate")
        parser.add_argument("--seed", type=int, default=42, help="Random seed for consistent evaluation samples")
        parser.add_argument("--use_entropy_control", action="store_true", help="Enable entropy-based early stopping and continuation")
        parser.add_argument("--entropy_threshold", type=float, default=5.0, help="Entropy threshold for early stopping")
        parser.add_argument("--max_retries", type=int, default=5, help="Maximum number of retries for entropy-controlled generation")
        parser.add_argument("--times", type=int, default=0, help="Number of optimization iterations")
        parser.add_argument("--lr", type=float, default=0.1, help="Learning rate for optimization")
        parser.add_argument("--record_entropy", action="store_true", help="Whether to record entropy analysis")
        parser.add_argument("--entropy_output_file", type=str, default="my_analysis.jsonl", help="Output file for entropy analysis")
        parser.add_argument("--entropy_weight", type=float, default=0.1, help="Weight for entropy loss")
        parser.add_argument("--adaptive_entropy", action="store_true", help="Enable adaptive entropy threshold")
        parser.add_argument("--adaptive_entropy_N", type=int, default=20, help="Number of samples for adaptive entropy threshold")
        parser.add_argument("--adaptive_entropy_K", type=float, default=2, help="K for adaptive entropy threshold")
        parser.add_argument("--mask_special_tokens", action="store_true", help="Mask special tokens in the input")
        
        # Parallel evaluation arguments
        parser.add_argument("--parallel", action="store_true", help="Enable parallel evaluation across multiple GPUs")
        parser.add_argument("--max_parallel_gpus", type=int, default=None, help="Maximum number of GPUs to use for parallel evaluation")
        
        # Average evaluation arguments
        parser.add_argument("--average", type=int, default=1, help="Number of times to run evaluation and take average")

        parser.add_argument("--version", type=str, help="Version of Same dataset")
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
        os.environ["adaptive_entropy"] = "True" if args.adaptive_entropy else "False"
        os.environ["adaptive_entropy_N"] = str(args.adaptive_entropy_N)
        os.environ["adaptive_entropy_K"] = str(args.adaptive_entropy_K)                                                                                     
        os.environ["temperature"] = str(args.temperature) if args.do_sample else "1.0"

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
        model_name = args.model_path.split("/")[-1]
        log_dir = f"logs/{benchmark_name}/{model_name}"
        os.makedirs(log_dir, exist_ok=True)
        max_retries = args.max_retries
        entropy_suffix = f"_entropy_{args.entropy_threshold}_weight_{args.entropy_weight}" if args.use_entropy_control else ""
        adaptive_entropy_suffix = f"_N_{args.adaptive_entropy_N}_K_{args.adaptive_entropy_K}" if args.adaptive_entropy else ""
        do_sample_suffix = f"_do_sample_temperature_{args.temperature}" if args.do_sample else ""

        mask_special_suffix = "" if args.mask_special_tokens else "_nomask"
        parallel_suffix = "_parallel" if getattr(args, 'parallel', False) else ""
        average_suffix = f"_avg_{args.average}" if args.average > 1 else ""

        log_file = os.path.join(log_dir, f"log_{model_name}_times_{args.times}_lr_{args.lr}{entropy_suffix}{adaptive_entropy_suffix}_reatries_{max_retries}{do_sample_suffix}{mask_special_suffix}{parallel_suffix}{average_suffix}.txt")
        
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
            f.write(f"Max Retries: {args.max_retries}\n")
            f.write(f"Parallel Evaluation: {getattr(args, 'parallel', False)}\n")
            if getattr(args, 'parallel', False):
                f.write(f"Max Parallel GPUs: {getattr(args, 'max_parallel_gpus', 'All available')}\n")
            f.write(f"Average Runs: {args.average}\n")
            f.write("\n")
        
        return log_file