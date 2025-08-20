#!/usr/bin/env python3
"""
Quick verification script to test TNOT decorator functionality
"""

import os
import sys

def quick_test():
    """Quick test of TNOT decorator import and basic functionality"""
    
    print("🔍 Quick TNOT Decorator Test")
    print("=" * 40)
    
    try:
        # Test 1: Import decorator
        print("1. Testing imports...")
        from TNOT.tnot_decorator import enable_tnot
        print("   ✅ Successfully imported enable_tnot")
        
        # Test 2: Import transformers
        from transformers import AutoModelForCausalLM
        print("   ✅ Successfully imported AutoModelForCausalLM")
        
        # Test 3: Apply decorator
        print("\n2. Testing decorator application...")
        TNOTModel = enable_tnot(AutoModelForCausalLM)
        print("   ✅ Successfully applied decorator to AutoModelForCausalLM")
        
        # Test 4: Check class attributes
        print("\n3. Checking enhanced class...")
        enhanced_methods = ['reset_entropy_detection', 'reset_model_parameters', '_safe_decode_token']
        for method in enhanced_methods:
            if hasattr(TNOTModel, method):
                print(f"   ✅ Method {method} added successfully")
            else:
                print(f"   ❌ Method {method} missing")
                return False
        
        # Test 5: Test BaseEvaluator import
        print("\n4. Testing BaseEvaluator integration...")
        from TNOT.base_evaluator import BaseEvaluator
        evaluator = BaseEvaluator()
        print("   ✅ BaseEvaluator imported and instantiated successfully")
        
        print("\n🎉 All quick tests passed!")
        print("\n📋 Next steps:")
        print("   • Run full test: python test_tnot_decorator.py")
        print("   • Run example: python example_usage.py")
        print("   • Check documentation: TNOT_DECORATOR_README.md")
        
        return True
        
    except ImportError as e:
        print(f"   ❌ Import error: {e}")
        print("\n💡 Make sure you're in the correct directory and have all dependencies installed")
        return False
        
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    sys.exit(0 if success else 1)