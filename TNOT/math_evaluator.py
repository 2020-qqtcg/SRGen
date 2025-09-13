import re
import random
import sympy
from sympy import Basic, MatrixBase, Float, Number, Rational, E, Symbol, Mul, simplify
from sympy.parsing.sympy_parser import parse_expr
from datasets import load_dataset
from TNOT.base_evaluator import BaseEvaluator
from functools import lru_cache
from itertools import product
from transformers import AutoTokenizer
import logging

class MATH500Evaluator(BaseEvaluator):
    def __init__(self):
        super().__init__()
        # Import latex2sympy2_extended if available
        try:
            from latex2sympy2_extended.latex2sympy2 import latex2sympy
            self.latex2sympy = latex2sympy
            self.has_latex2sympy = True
        except ImportError:
            self.has_latex2sympy = False
            logging.warning("latex2sympy2_extended not available. LaTeX parsing will be limited.")

    def load_dataset(self, split="test", eval_samples=None, version="math500", **kwargs):
        """Load MATH500 dataset"""
        try:
            if version == "math500":
                eval_dataset = load_dataset("HuggingFaceH4/MATH-500", split=split)
            elif version == "amc":
                eval_dataset = load_dataset("AI-MO/NuminaMath-CoT", split=split)
            elif version == "hmmt2025":
                eval_dataset = load_dataset("MathArena/hmmt_feb_2025", split=split)
            else:
                eval_dataset = []
        except:
            raise ValueError("MATH dataset not found. Please ensure the dataset is available.")
        
        eval_QAs = []
        for item in eval_dataset:
            if 'problem' in item and 'solution' in item:
                question = item['problem']
                answer = item['solution']
            elif 'problem' in item and 'answer' in item:
                question = item['problem']
                answer = item['answer']
            elif 'question' in item and 'answer' in item:
                question = item['question']
                answer = item['answer']
            else:
                continue
                
            eval_QAs.append({'Q': question, 'A': answer})
        
        if eval_samples is not None and len(eval_QAs) > eval_samples:
            eval_QAs = random.sample(eval_QAs, eval_samples)
        
        # new_eval_QAs = []
        # for idx, item in enumerate(eval_QAs):
        #     if idx in [9, 11, 18, 23, 25, 31, 36, 67, 88, 94, 100, 106, 119, 154, 157, 166, 189, 204, 217, 235, 239, 240, 246, 257, 264, 284, 286, 301, 303, 305, 308, 324, 340, 362, 379, 383, 393, 400, 422, 425, 444, 460, 467, 490]:
        #         new_eval_QAs.append(item)
        return eval_QAs
        # return new_eval_QAs

    def get_latex_regex_patterns(self):
        """Get LaTeX extraction regex patterns with priorities (lower = higher priority)"""
        patterns = [
            # LightEval specific patterns - highest priority
            (r"(?i:therefore|so),?\s*the\s*final\s*answer\s*is:\s*\$\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\$", 0),
            (r"(?i:final\s*answer)\s*is:\s*\$\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}\$", 5),
            
            # Standard boxed patterns
            (r"\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}", 10),
            
            # Answer with colon patterns
            (r"(?i:answer)\s*:\s*\$([^$]+)\$", 20),
            (r"(?i:answer)\s*:\s*([^.\n,;]+)", 25),
            
            # Math mode patterns
            (r"(?i:final\s*answer|the\s*answer\s*is)\s*:?\s*\$([^$]+)\$", 30),
            (r"(?i:therefore|thus|so)\s*,?\s*\$([^$]+)\$", 35),
            
            # LaTeX environments
            (r"\\\[([^\]]+)\\\]", 40),
            (r"\$\$([^$]+)\$\$", 45),
        ]
        return patterns

    def get_expr_regex_patterns(self):
        """Get mathematical expression regex patterns"""
        # Basic number patterns
        number_re = (
            r"(?:"
            r"(?P<integer1>-?\d{1,3}(?:[ ,]\d{3})+)(?P<decimal1>\.\d+)?|"  # 1,234.56
            r"(?P<integer2>-?\d+)(?P<decimal2>[.,]\d+)|"                    # 123.45 or 123,45
            r"(?P<decimal3>\.\d+)|"                                         # .123
            r"(?P<integer3>-?\d+)"                                          # 123
            r")(?P<percent>\s*(?:%|[Pp]ercent|\s*[Pp]ercentage|\s*[Pp]ct))?"
        )
        
        # Expressions with operators
        operators = [r"\+", r"\-", r"\*", r"\×", r"\/", r"\^", r"\(", r"\)", r"\÷"]
        operators_re = "".join(operators)
        all_expr_chars = r"[\d\.\s" + operators_re + r"]"
        expr_re = rf"(?P<expr>-?\(?-?\d{all_expr_chars}*[{operators_re}]{all_expr_chars}+\)?)"
        
        patterns = [
            # Answer prefixed patterns
            (rf"(?i:answer)\s*:\s*{number_re}", 10),
            (rf"(?i:answer)\s*:\s*{expr_re}", 15),
            (rf"(?i:final\s*answer)\s*:?\s*{number_re}", 20),
            (rf"(?i:final\s*answer)\s*:?\s*{expr_re}", 25),
            
            # Plain patterns (lower priority)
            (number_re, 50),
            (expr_re, 55),
        ]
        return patterns

    def extract_boxed_content(self, text, start_pos):
        """Extract content from \\boxed{...} handling nested braces correctly"""
        brace_start = text.find('{', start_pos)
        if brace_start == -1:
            return None
        
        brace_count = 1
        pos = brace_start + 1
        
        while pos < len(text) and brace_count > 0:
            if text[pos] == '{':
                brace_count += 1
            elif text[pos] == '}':
                brace_count -= 1
            pos += 1
        
        if brace_count == 0:
            content = text[brace_start + 1:pos - 1]
            return content.strip()
        
        return None

    def extract_latex_answer(self, text):
        """Extract LaTeX answers from text using lighteval-style patterns"""
        best_match = None
        best_priority = float('inf')
        
        # First try boxed patterns with proper brace matching
        boxed_patterns = [
            r"(?i:therefore|so),?\s*the\s*final\s*answer\s*is:\s*\$\\boxed",
            r"(?i:final\s*answer)\s*is:\s*\$\\boxed",
            r"\\boxed",
        ]
        
        for i, pattern in enumerate(boxed_patterns):
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.DOTALL))
            if matches:
                for match in matches:
                    content = self.extract_boxed_content(text, match.end())
                    if content is not None:
                        priority = i * 10
                        if priority < best_priority:
                            best_priority = priority
                            best_match = content
        
        if best_match is not None:
            return best_match
        
        # Try other LaTeX patterns
        for pattern, priority in self.get_latex_regex_patterns():
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            if matches and priority < best_priority:
                match = matches[-1]  # Take the last match
                best_priority = priority
                best_match = match.group(1) if match.groups() else match.group(0)
        
        return best_match

    def extract_expr_answer(self, text):
        """Extract mathematical expressions from text"""
        best_match = None
        best_priority = float('inf')
        
        for pattern, priority in self.get_expr_regex_patterns():
            matches = list(re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE))
            if matches and priority < best_priority:
                match = matches[-1]  # Take the last match
                best_priority = priority
                
                # Handle different group patterns
                if match.groups():
                    # Find the first non-None group
                    for group in match.groups():
                        if group is not None:
                            best_match = group
                            break
                else:
                    best_match = match.group(0)
        
        return best_match

    def extract_answer_comprehensive(self, text):
        """Comprehensive answer extraction using lighteval-style approach"""
        if not text:
            return None
        
        # First try LaTeX extraction (higher priority)
        latex_answer = self.extract_latex_answer(text)
        if latex_answer:
            return latex_answer
        
        # Then try expression extraction
        expr_answer = self.extract_expr_answer(text)
        if expr_answer:
            return expr_answer
        
        return None

    def parse_latex_to_sympy(self, latex_str):
        """Parse LaTeX string to SymPy expression with timeout"""
        if not self.has_latex2sympy:
            return None, latex_str
        
        try:
            # Clean up the LaTeX string
            latex_str = latex_str.strip()
            
            # Handle special cases
            if latex_str.lower() in ['true', 'false']:
                return latex_str.lower(), latex_str
            
            # Try to parse with latex2sympy
            sympy_expr = self.latex2sympy(latex_str)
            return sympy_expr, latex_str
        except:
            return None, latex_str

    def parse_expr_to_sympy(self, expr_str):
        """Parse mathematical expression to SymPy with timeout"""
        try:
            expr_str = expr_str.strip()
            
            # Handle percentages
            if expr_str.endswith('%'):
                number_part = expr_str[:-1].strip()
                number = parse_expr(number_part, evaluate=False)
                return sympy.Mul(number, sympy.Rational(1, 100), evaluate=False)
            
            # Parse regular expression
            return parse_expr(expr_str, evaluate=False)
        except:
            return None

    def safe_sympy_doit(self, expr):
        """Safely evaluate SymPy expression"""
        try:
            return expr.doit() if hasattr(expr, 'doit') else expr
        except:
            return expr

    def sympy_numeric_eq(self, a, b, precision=6):
        """Compare two SymPy expressions numerically"""
        try:
            # Handle matrix expressions
            if isinstance(a, (MatrixBase,)) and isinstance(b, (MatrixBase,)):
                a = self.safe_sympy_doit(a)
                b = self.safe_sympy_doit(b)
                if hasattr(a, 'shape') and hasattr(b, 'shape') and a.shape == b.shape:
                    return all(self.sympy_numeric_eq(a_elem, b_elem, precision) 
                             for a_elem, b_elem in zip(a.flat(), b.flat()))
            
            # Handle numbers with precision
            if isinstance(a, Number) or isinstance(b, Number):
                if isinstance(a, Float) or isinstance(b, Float):
                    a = self.safe_sympy_doit(a)
                    b = self.safe_sympy_doit(b)
                    if isinstance(a, Number) and isinstance(b, Number):
                        return a.round(precision) == b.round(precision)
                else:
                    return self.safe_sympy_doit(a) == self.safe_sympy_doit(b)
            
            # Try numerical comparison
            try:
                return (a - b).evalf(chop=True) == 0
            except:
                pass
        except:
            pass
        
        return False

    def sympy_symbolic_eq(self, a, b):
        """Compare two SymPy expressions symbolically"""
        try:
            # Direct equality
            if a == b:
                return True
            
            # Try simplification
            try:
                diff = simplify(a - b)
                return diff == 0
            except:
                pass
            
            # Try string comparison after sorting
            try:
                return str(a) == str(b)
            except:
                pass
        except:
            pass
        
        return False

    def sympy_compare_symbols(self, a, b):
        """Compare symbols with special handling for E and concatenated symbols"""
        try:
            # Handle E vs symbol case
            if (isinstance(a, Symbol) and a.name.lower() == "e" and b == E) or \
               (isinstance(b, Symbol) and b.name.lower() == "e" and a == E):
                return True
            
            # Handle multiplication of symbols vs single symbol
            if isinstance(a, Symbol) and isinstance(b, Mul):
                if all(arg == E or isinstance(arg, Symbol) for arg in b.args):
                    concat_b = "".join(arg.name if isinstance(arg, Symbol) else "e" for arg in b.args)
                    return a.name.lower() == concat_b.lower()
            
            if isinstance(b, Symbol) and isinstance(a, Mul):
                if all(arg == E or isinstance(arg, Symbol) for arg in a.args):
                    concat_a = "".join(arg.name if isinstance(arg, Symbol) else "e" for arg in a.args)
                    return b.name.lower() == concat_a.lower()
            
            return a == b
        except:
            return False

    def sympy_expr_eq(self, gold, pred, precision=6):
        """Compare two SymPy expressions using multiple methods"""
        try:
            # Quick string comparison
            if str(gold) == str(pred):
                return True
            
            # Direct equality
            if gold == pred:
                return True
            
            # Symbol comparison
            if isinstance(gold, Symbol) or isinstance(pred, Symbol):
                return self.sympy_compare_symbols(gold, pred)
            
            # Numeric comparison
            if self.sympy_numeric_eq(gold, pred, precision):
                return True
            
            # Symbolic comparison
            if self.sympy_symbolic_eq(gold, pred):
                return True
            
        except:
            pass
        
        return False

    def compare_extracted_answers(self, gold_list, pred_list, precision=6, timeout_seconds=3):
        """Compare lists of extracted answers using SymPy"""
        def compare_single(gold, pred):
            # If both are SymPy expressions
            if isinstance(gold, (Basic, MatrixBase)) and isinstance(pred, (Basic, MatrixBase)):
                return self.sympy_expr_eq(gold, pred, precision)
            
            # If both are strings
            elif isinstance(gold, str) and isinstance(pred, str):
                gold = gold.strip()
                pred = pred.strip()
                return len(gold) > 0 and len(pred) > 0 and gold.lower() == pred.lower()
            
            return False
        
        def safe_compare(g, p):
            try:
                return compare_single(g, p)
            except:
                return False
        
        # Try all combinations
        return any(safe_compare(g, p) for g, p in product(gold_list, pred_list))

    def reward_correct(self, item, answer):
        """Check if the answer is correct using lighteval-style evaluation"""
        # Extract predicted answer
        predicted_raw = self.extract_answer_comprehensive(answer)
        if predicted_raw is None:
            return -1.0
        
        # Extract ground truth
        ground_truth = item.get("A", "").strip()
        ground_truth_raw = self.extract_answer_comprehensive(ground_truth)
        if ground_truth_raw is None:
            ground_truth_raw = ground_truth
        
        # Parse both to SymPy expressions
        pred_parsed_list = []
        gold_parsed_list = []
        
        # Try LaTeX parsing first
        try:
            pred_sympy, pred_str = self.parse_latex_to_sympy(predicted_raw)
            if pred_sympy is not None:
                pred_parsed_list.append(pred_sympy)
            else:
                pred_parsed_list.append(pred_str)
        except:
            # Fallback to expression parsing
            try:
                pred_expr = self.parse_expr_to_sympy(predicted_raw)
                if pred_expr is not None:
                    pred_parsed_list.append(pred_expr)
                else:
                    pred_parsed_list.append(predicted_raw)
            except:
                pred_parsed_list.append(predicted_raw)
        
        # Parse ground truth
        try:
            gold_sympy, gold_str = self.parse_latex_to_sympy(ground_truth_raw)
            if gold_sympy is not None:
                gold_parsed_list.append(gold_sympy)
            else:
                gold_parsed_list.append(gold_str)
        except:
            try:
                gold_expr = self.parse_expr_to_sympy(ground_truth_raw)
                if gold_expr is not None:
                    gold_parsed_list.append(gold_expr)
                else:
                    gold_parsed_list.append(ground_truth_raw)
            except:
                gold_parsed_list.append(ground_truth_raw)
        
        # Compare using lighteval-style comparison
        if self.compare_extracted_answers(gold_parsed_list, pred_parsed_list, precision=6):
            return 1.0
        
        # Fallback string comparison
        try:
            if predicted_raw.strip().lower() == ground_truth_raw.strip().lower():
                return 1.0
        except:
            pass
        
        return -1.0

    def get_system_prompt(self):
        """Get system prompt for MATH500"""
        return """Solve the following math problem efficiently and clearly. The last line of your response should be of the following format: 'Therefore, the final answer is: $\\boxed{{ANSWER}}$. I hope it is correct' (without quotes) where ANSWER is just the final number or expression that solves the problem. Think step by step before answering."""

    def setup_logging(self, args):
        return super().setup_logging(args, benchmark_name=args.version)

def main():
    # Initialize evaluator
    evaluator = MATH500Evaluator()
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
                    max_parallel_gpus=args.max_parallel_gpus,
                    version=args.version
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
                max_parallel_gpus=args.max_parallel_gpus,
                version=args.version
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
