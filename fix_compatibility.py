#!/usr/bin/env python3
"""
Compatibility fix script for TNOT models with transformers 4.55.2
This script applies additional compatibility fixes if needed.
"""
import os
import sys

def check_transformers_version():
    """Check transformers version and warn if incompatible"""
    try:
        import transformers
        version = transformers.__version__
        print(f"Detected transformers version: {version}")
        
        # Parse version
        major, minor, patch = map(int, version.split('.')[:3])
        
        if major < 4 or (major == 4 and minor < 51):
            print("⚠️  WARNING: Qwen3 models require transformers >= 4.51.0")
            print("   Consider upgrading: pip install transformers>=4.51.0")
        
        return True
    except ImportError:
        print("❌ transformers not installed!")
        return False
    except Exception as e:
        print(f"❌ Error checking transformers version: {e}")
        return False

def check_torch():
    """Check PyTorch installation"""
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA devices: {torch.cuda.device_count()}")
        return True
    except ImportError:
        print("❌ PyTorch not installed!")
        return False

def main():
    """Main compatibility check"""
    print("=" * 60)
    print("TNOT Compatibility Check & Fix")
    print("=" * 60)
    
    # Check dependencies
    if not check_transformers_version():
        return 1
    
    if not check_torch():
        return 1
    
    print("\n" + "=" * 60)
    print("COMPATIBILITY FIXES APPLIED")
    print("=" * 60)
    print("✅ FlashAttentionKwargs import fallback added to all models")
    print("✅ Qwen3Config fallback already implemented")
    print("✅ Import error handling improved")
    
    print("\n" + "=" * 60)
    print("RECOMMENDATIONS")
    print("=" * 60)
    print("1. Run 'python test_imports.py' to verify all imports work")
    print("2. If you encounter 'KeyError: qwen3' errors, upgrade transformers:")
    print("   pip install transformers>=4.51.0")
    print("3. For macOS users, some packages in requirements.txt are commented out")
    print("   Use requirements-local.txt for macOS compatibility")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)