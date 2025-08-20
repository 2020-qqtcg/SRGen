#!/usr/bin/env python3
"""
Example usage of the TNOT decorator

This script demonstrates how to use the universal TNOT decorator 
to enable TNOT functionality on any Transformers CausalLM model.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from TNOT.tnot_decorator import enable_tnot

def main():
    """Main example function"""
    
    print("TNOT Universal Decorator Example")
    print("=" * 50)
    
    # Example 1: Using with AutoModelForCausalLM
    print("\n1. Creating TNOT-enabled AutoModelForCausalLM")
    
    # Apply decorator to create TNOT-enabled class
    TNOTAutoModel = enable_tnot(AutoModelForCausalLM)
    
    # You can now use this class like any other model class
    print("   ✓ TNOT decorator applied to AutoModelForCausalLM")
    
    # Example 2: Loading a specific model (replace with your model path)
    model_path = "microsoft/DialoGPT-small"  # Example model
    print(f"\n2. Loading model: {model_path}")
    
    try:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Load model with TNOT capabilities
        model = TNOTAutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.float32,
            device_map="cpu"  # Use CPU for this example
        )
        
        print("   ✓ Model loaded successfully with TNOT capabilities")
        
        # Verify TNOT attributes are present
        tnot_attributes = ['delta', 'high_entropy_detected', 'entropy_history', 
                          'reset_entropy_detection', 'reset_model_parameters']
        
        for attr in tnot_attributes:
            if hasattr(model, attr):
                print(f"   ✓ {attr} attribute/method present")
            else:
                print(f"   ✗ {attr} attribute/method missing")
        
        # Example 3: Basic usage with TNOT settings
        print("\n3. Testing TNOT functionality")
        
        # Set TNOT environment variables
        os.environ["prompt_only"] = "True"
        os.environ["times"] = "1"
        os.environ["lr"] = "0.1"
        os.environ["entropy_weight"] = "0.1"
        
        # Test input
        test_text = "Hello, how are you?"
        inputs = tokenizer(test_text, return_tensors="pt")
        
        print(f"   Input: {test_text}")
        print(f"   Input shape: {inputs.input_ids.shape}")
        
        # Forward pass (prompt stage)
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"   ✓ Prompt stage completed, output shape: {outputs.logits.shape}")
        
        # Switch to generation stage
        os.environ["prompt_only"] = "False"
        
        # Forward pass (generation stage)
        with torch.no_grad():
            outputs = model(**inputs)
        
        print(f"   ✓ Generation stage completed, output shape: {outputs.logits.shape}")
        
        print("\n4. Testing utility methods")
        
        # Test reset methods
        model.reset_entropy_detection()
        print("   ✓ reset_entropy_detection() called")
        
        model.reset_model_parameters()
        print("   ✓ reset_model_parameters() called")
        
        print("\n🎉 All tests completed successfully!")
        print("\nThe TNOT decorator is working correctly and can be used as a drop-in replacement")
        print("for the individual modeling files.")
        
    except Exception as e:
        print(f"\n❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up environment variables
        env_vars = ["prompt_only", "times", "lr", "entropy_weight"]
        for var in env_vars:
            if var in os.environ:
                del os.environ[var]


def demonstrate_integration():
    """Demonstrate how the decorator integrates with existing code"""
    
    print("\n" + "=" * 50)
    print("Integration Example")
    print("=" * 50)
    
    print("\nBefore (using specific modeling files):")
    print("""
    from TNOT.model.modeling_llama3_tnot import LlamaForCausalLM
    from TNOT.model.modeling_phi3_tnot import Phi3ForCausalLM
    from TNOT.model.modeling_qwen2_tnot import Qwen2ForCausalLM
    
    # Need different classes for different models
    if model_type == "llama":
        model = LlamaForCausalLM.from_pretrained(model_path)
    elif model_type == "phi3":
        model = Phi3ForCausalLM.from_pretrained(model_path)
    elif model_type == "qwen2":
        model = Qwen2ForCausalLM.from_pretrained(model_path)
    """)
    
    print("\nAfter (using universal decorator):")
    print("""
    from transformers import AutoModelForCausalLM
    from TNOT.tnot_decorator import enable_tnot
    
    # One class works for all models!
    TNOTModel = enable_tnot(AutoModelForCausalLM)
    model = TNOTModel.from_pretrained(model_path)
    """)
    
    print("\n✨ Benefits:")
    print("   • Unified interface for all model types")
    print("   • No need for separate modeling files")
    print("   • Easier maintenance and updates")
    print("   • Automatic support for new model types")
    print("   • Same TNOT functionality across all models")


if __name__ == "__main__":
    main()
    demonstrate_integration()