#!/usr/bin/env python3
"""
Test script to verify TNOT model imports work correctly with transformers 4.55.2
"""
import sys
import traceback

def test_import(module_name, description):
    """Test importing a module and report results"""
    print(f"Testing {description}...")
    try:
        if module_name == "base_evaluator":
            from TNOT.base_evaluator import BaseEvaluator
            print(f"✅ {description} - SUCCESS")
            return True
        elif module_name == "qwen2_tnot":
            from TNOT.model.modeling_qwen2_tnot import Qwen2ForCausalLM
            print(f"✅ {description} - SUCCESS")
            return True
        elif module_name == "llama3_tnot":
            from TNOT.model.modeling_llama3_tnot import LlamaForCausalLM
            print(f"✅ {description} - SUCCESS")
            return True
        elif module_name == "phi3_tnot":
            from TNOT.model.modeling_phi3_tnot import Phi3ForCausalLM
            print(f"✅ {description} - SUCCESS")
            return True
        elif module_name == "qwen3_tnot":
            from TNOT.model.modeling_qwen3_tnot import Qwen3ForCausalLM
            print(f"✅ {description} - SUCCESS")
            return True
        else:
            print(f"❓ Unknown module: {module_name}")
            return False
    except Exception as e:
        print(f"❌ {description} - FAILED")
        print(f"   Error: {str(e)}")
        print(f"   Traceback: {traceback.format_exc()}")
        return False

def main():
    """Main test function"""
    print("=" * 60)
    print("TNOT Model Import Compatibility Test")
    print("=" * 60)
    print(f"Python version: {sys.version}")
    
    # Test transformers version
    try:
        import transformers
        print(f"Transformers version: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed!")
        return
    
    print("=" * 60)
    
    # List of tests to run
    tests = [
        ("base_evaluator", "BaseEvaluator"),
        ("qwen2_tnot", "Qwen2 TNOT Model"),
        ("llama3_tnot", "LLaMA3 TNOT Model"),
        ("phi3_tnot", "Phi3 TNOT Model"),
        ("qwen3_tnot", "Qwen3 TNOT Model"),
    ]
    
    results = []
    for module_name, description in tests:
        success = test_import(module_name, description)
        results.append((description, success))
        print()
    
    # Summary
    print("=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for description, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} - {description}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All imports working correctly!")
        return 0
    else:
        print("⚠️  Some imports failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)