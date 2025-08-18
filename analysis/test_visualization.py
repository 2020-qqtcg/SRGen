#!/usr/bin/env python3
"""
Test script for the response entropy visualization tool
"""

import json
import os
import sys
from pathlib import Path

def create_test_data():
    """Create test entropy data"""
    test_data = [
        {
            "token_index": 0,
            "original_entropy": 2.34,
            "modified_entropy": 3.12,
            "original_token_decoded": "The",
            "modified_token_decoded": "A"
        },
        {
            "token_index": 1,
            "original_entropy": 1.87,
            "modified_entropy": 1.45,
            "original_token_decoded": "quick",
            "modified_token_decoded": "fast"
        },
        {
            "token_index": 2,
            "original_entropy": 3.21,
            "modified_entropy": 2.98,
            "original_token_decoded": "brown",
            "modified_token_decoded": "brown"
        },
        {
            "token_index": 3,
            "original_entropy": 2.56,
            "modified_entropy": 3.45,
            "original_token_decoded": "fox",
            "modified_token_decoded": "dog"
        },
        {
            "token_index": 4,
            "original_entropy": 1.23,
            "modified_entropy": 1.67,
            "original_token_decoded": "jumps",
            "modified_token_decoded": "runs"
        }
    ]
    
    test_file = "test_entropy_data.json"
    with open(test_file, 'w') as f:
        json.dump(test_data, f, indent=2)
    
    print(f"Created test data file: {test_file}")
    return test_file

def test_visualization():
    """Test the visualization tool"""
    try:
        # Import the visualization functions
        from analyze_response import load_entropy_data, extract_entropy_data, create_entropy_plot
        
        # Create test data
        test_file = create_test_data()
        
        # Load data
        data = load_entropy_data(test_file)
        if data is None:
            print("❌ Failed to load test data")
            return False
        
        # Extract data
        indices, original_entropy, modified_entropy, original_tokens, modified_tokens = extract_entropy_data(data)
        
        if not indices:
            print("❌ Failed to extract data")
            return False
        
        print(f"✅ Successfully loaded {len(indices)} records")
        print(f"   Token indices: {indices}")
        print(f"   Original entropy range: {min(original_entropy):.2f} - {max(original_entropy):.2f}")
        print(f"   Modified entropy range: {min(modified_entropy):.2f} - {max(modified_entropy):.2f}")
        
        # Test plot creation
        try:
            create_entropy_plot(indices, original_entropy, modified_entropy, 
                              original_tokens, modified_tokens, 
                              output_path="test_plot.png")
            print("✅ Successfully created test plot: test_plot.png")
        except Exception as e:
            print(f"❌ Failed to create plot: {e}")
            return False
        
        # Clean up test files
        if os.path.exists(test_file):
            os.remove(test_file)
            print(f"✅ Cleaned up test file: {test_file}")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you have installed the required dependencies:")
        print("  pip install matplotlib numpy seaborn")
        return False
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False

def main():
    print("Testing Response Entropy Visualization Tool")
    print("=" * 50)
    
    success = test_visualization()
    
    if success:
        print("\n✅ All tests passed!")
        print("The visualization tool is working correctly.")
        print("\nYou can now use it with your own data:")
        print("  python analyze_response.py your_entropy_data.json")
    else:
        print("\n❌ Tests failed!")
        print("Please check the error messages above.")

if __name__ == "__main__":
    main() 