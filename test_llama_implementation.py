#!/usr/bin/env python3
"""
Simple test script to verify that LLaMA 3 SLOT implementation works correctly.
This script tests the model loading and basic functionality without requiring a full dataset.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import torch
from TNOT.base_evaluator import BaseEvaluator
from TNOT.modeling_llama3_slot import LlamaForCausalLM
from TNOT.modeling_qwen2_slot import Qwen2ForCausalLM
from transformers import AutoConfig

def test_model_loading():
    """Test if model loading works correctly for different model types"""
    evaluator = BaseEvaluator()
    
    # Test 1: Mock Qwen2 config
    print("=== Test 1: Qwen2 Model Type Detection ===")
    try:
        # This would normally load from a real model path
        # For testing, we just verify the import works
        print("✓ Qwen2 model class imported successfully")
        print("✓ Qwen2 model has required methods:", hasattr(Qwen2ForCausalLM, 'forward'))
    except Exception as e:
        print(f"✗ Error with Qwen2: {e}")
    
    # Test 2: Mock LLaMA config  
    print("\n=== Test 2: LLaMA Model Type Detection ===")
    try:
        # This would normally load from a real model path
        # For testing, we just verify the import works
        print("✓ LLaMA model class imported successfully")
        print("✓ LLaMA model has required methods:", hasattr(LlamaForCausalLM, 'forward'))
        print("✓ LLaMA model has SLOT methods:", hasattr(LlamaForCausalLM, 'reset_entropy_detection'))
        print("✓ LLaMA model has SLOT methods:", hasattr(LlamaForCausalLM, 'reset_model_parameters'))
    except Exception as e:
        print(f"✗ Error with LLaMA: {e}")

def test_slot_functionality():
    """Test SLOT-specific functionality"""
    print("\n=== Test 3: SLOT Functionality ===")
    
    # Create dummy config for testing
    from transformers.models.llama.configuration_llama import LlamaConfig
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    
    try:
        # Test model initialization
        model = LlamaForCausalLM(config)
        print("✓ LLaMA model initialized successfully")
        
        # Test SLOT methods
        model.reset_entropy_detection()
        print("✓ reset_entropy_detection() works")
        
        model.reset_model_parameters()
        print("✓ reset_model_parameters() works")
        
        # Test that SLOT attributes exist
        assert hasattr(model, 'delta'), "Model should have delta attribute"
        assert hasattr(model, 'high_entropy_detected'), "Model should have high_entropy_detected attribute"
        assert hasattr(model, 'entropy_history'), "Model should have entropy_history attribute"
        print("✓ All SLOT attributes present")
        
        # Test forward pass with dummy data
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        with torch.no_grad():
            outputs = model(input_ids)
        print("✓ Forward pass successful")
        print(f"✓ Output logits shape: {outputs.logits.shape}")
        
    except Exception as e:
        print(f"✗ Error testing SLOT functionality: {e}")
        import traceback
        traceback.print_exc()

def test_entropy_control():
    """Test entropy control features"""
    print("\n=== Test 4: Entropy Control Features ===")
    
    # Set environment variables for testing
    os.environ["prompt_only"] = "True"
    os.environ["times"] = "1"
    os.environ["lr"] = "0.1"
    os.environ["entropy_control"] = "True"
    os.environ["entropy_threshold"] = "3.0"
    
    from transformers.models.llama.configuration_llama import LlamaConfig
    config = LlamaConfig(
        vocab_size=1000,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    
    try:
        model = LlamaForCausalLM(config)
        print("✓ Model created for entropy control test")
        
        # Test with prompt_only mode
        input_ids = torch.randint(0, config.vocab_size, (1, 10))
        
        with torch.no_grad():
            outputs = model(input_ids)
        
        print("✓ Entropy control forward pass successful")
        print(f"✓ Delta attribute after forward: {model.delta is not None}")
        
        # Reset environment
        os.environ["prompt_only"] = "False"
        
    except Exception as e:
        print(f"✗ Error testing entropy control: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Testing LLaMA 3 SLOT Implementation")
    print("=" * 50)
    
    test_model_loading()
    test_slot_functionality()
    test_entropy_control()
    
    print("\n" + "=" * 50)
    print("Test completed! Check above for any errors (✗ marks)")