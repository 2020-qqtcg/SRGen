# TNOT Universal Decorator

This document describes the universal TNOT (Test-time Training) decorator that enables TNOT functionality on any Transformers CausalLM model without requiring separate modeling files.

## Overview

The TNOT decorator (`tnot_decorator.py`) provides a unified way to add TNOT capabilities to any Transformers model, eliminating the need for separate modeling files for each model type (LLaMA, Phi3, Qwen2, etc.).

## Key Features

- **Universal Compatibility**: Works with any Transformers CausalLM model
- **Drop-in Replacement**: Can replace existing TNOT modeling files
- **Identical Functionality**: Provides the same TNOT features as the original implementations
- **Easy Integration**: Simple decorator-based API
- **Maintenance Friendly**: Single file to maintain instead of multiple modeling files

## Usage

### Basic Usage

```python
from transformers import AutoModelForCausalLM
from TNOT.tnot_decorator import enable_tnot

# Create TNOT-enabled model class
TNOTModel = enable_tnot(AutoModelForCausalLM)

# Load any model with TNOT capabilities
model = TNOTModel.from_pretrained("your-model-path")
```

### Integration with BaseEvaluator

The `base_evaluator.py` has been updated to use the universal decorator:

```python
# Old approach (multiple specific classes)
from TNOT.model.modeling_llama3_tnot import LlamaForCausalLM
from TNOT.model.modeling_phi3_tnot import Phi3ForCausalLM
# ... etc

# New approach (universal decorator)
from TNOT.tnot_decorator import enable_tnot
TNOTModelClass = enable_tnot(AutoModelForCausalLM)
```

### Environment Variables

The decorator respects the same environment variables as the original implementation:

```python
import os

# TNOT Configuration
os.environ["prompt_only"] = "True"        # Enable prompt-stage optimization
os.environ["times"] = "5"                 # Number of optimization steps
os.environ["lr"] = "0.1"                  # Learning rate
os.environ["entropy_weight"] = "0.1"      # Weight for entropy loss

# Entropy Control
os.environ["entropy_control"] = "True"    # Enable entropy-based early stopping
os.environ["entropy_threshold"] = "3.0"   # Static entropy threshold
os.environ["adaptive_entropy"] = "True"   # Enable adaptive threshold
os.environ["adaptive_entropy_N"] = "20"   # Window size for adaptive threshold
os.environ["adaptive_entropy_K"] = "2.0"  # Multiplier for adaptive threshold

# Logging and Analysis
os.environ["record_entropy"] = "True"     # Enable entropy recording
os.environ["entropy_output_file"] = "entropy_analysis.jsonl"
os.environ["response_entropy_file"] = "response_entropy.json"
os.environ["log_entropy_control"] = "True"  # Enable entropy control logging
```

## API Reference

### Decorator Function

```python
enable_tnot(model_class)
```

**Parameters:**
- `model_class`: A Transformers CausalLM model class (e.g., `AutoModelForCausalLM`)

**Returns:**
- Enhanced model class with TNOT capabilities

### Added Methods and Attributes

The decorator adds the following to the model:

#### Attributes
- `delta`: Learned delta parameters for hidden state modification
- `high_entropy_detected`: Boolean flag for entropy detection
- `high_entropy_position`: Position where high entropy was detected
- `entropy_threshold`: Current entropy threshold
- `entropy_history`: History of entropy values
- `index`: Token index for response entropy tracking

#### Methods
- `reset_entropy_detection()`: Reset entropy detection state
- `reset_model_parameters()`: Reset all TNOT parameters
- `_safe_decode_token(token_id)`: Safely decode a token ID
- `_safe_decode_sequence(token_ids)`: Safely decode a token sequence

## Implementation Details

### TNOT Process Flow

1. **Prompt Stage** (`prompt_only=True`):
   - If `delta` exists: Optimize `delta_high` with CE + entropy loss
   - If `delta` is None: Optimize `delta_normal` with CE loss only
   - Store optimized delta for generation stage

2. **Generation Stage** (`prompt_only=False`):
   - Apply stored delta (currently commented out in original implementation)
   - Perform entropy-based early stopping if enabled

### Entropy Control

The decorator implements the same entropy control mechanism:
- Calculate entropy for each generated token
- Compare against static or adaptive threshold
- Force EOS token for high-entropy samples
- Log entropy events if enabled

### Compatibility

The decorator maintains full compatibility with:
- All Transformers model architectures
- Original TNOT parameter settings
- Existing evaluation scripts
- Entropy analysis and logging features

## Migration Guide

### From Specific Modeling Files

**Before:**
```python
from TNOT.model.modeling_llama3_tnot import LlamaForCausalLM
model = LlamaForCausalLM.from_pretrained(model_path)
```

**After:**
```python
from TNOT.tnot_decorator import enable_tnot
from transformers import AutoModelForCausalLM

TNOTModel = enable_tnot(AutoModelForCausalLM)
model = TNOTModel.from_pretrained(model_path)
```

### Updating BaseEvaluator

The `base_evaluator.py` has been automatically updated to use the decorator. No changes needed in evaluation scripts.

## Testing

Run the test script to verify functionality:

```bash
python test_tnot_decorator.py
```

Run the example usage script:

```bash
python example_usage.py
```

## Benefits

1. **Reduced Code Duplication**: Single implementation instead of multiple modeling files
2. **Easier Maintenance**: Updates only need to be made in one place
3. **Better Extensibility**: Automatically supports new model architectures
4. **Simplified Integration**: Drop-in replacement for existing code
5. **Consistent Behavior**: Identical TNOT functionality across all models

## File Structure

```
TNOT/
├── tnot_decorator.py           # Universal TNOT decorator
├── base_evaluator.py           # Updated to use decorator
├── test_tnot_decorator.py      # Test script
├── example_usage.py            # Usage examples
├── TNOT_DECORATOR_README.md    # This documentation
└── model/                      # Original modeling files (can be deprecated)
    ├── modeling_llama3_tnot.py
    ├── modeling_phi3_tnot.py
    ├── modeling_qwen2_tnot.py
    └── modeling_qwen3_tnot.py
```

## Troubleshooting

### Common Issues

1. **Missing Environment Variables**: Ensure required environment variables are set
2. **Model Loading Errors**: Check model path and permissions
3. **Memory Issues**: Use appropriate `torch_dtype` and `device_map`
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

Enable detailed logging:

```python
os.environ["log_entropy_control"] = "True"
```

### Performance Considerations

- Use `torch.bfloat16` for better memory efficiency
- Use appropriate `device_map` for multi-GPU setups
- Consider `_attn_implementation="flash_attention_2"` for supported models

## Future Enhancements

- [ ] Support for additional model architectures
- [ ] Performance optimizations
- [ ] Enhanced logging and debugging features
- [ ] Integration with more evaluation frameworks