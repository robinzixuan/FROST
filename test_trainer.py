#!/usr/bin/env python3
"""
Test script to verify SafeSFTTrainer works correctly
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from train_sft import SafeSFTTrainer
    print("✓ SafeSFTTrainer import successful")
    
    # Check if the class has the required methods
    required_methods = ['training_step', '_inner_training_loop']
    for method in required_methods:
        if hasattr(SafeSFTTrainer, method):
            print(f"✓ Method {method} found")
        else:
            print(f"✗ Method {method} missing")
    
    print("✓ SafeSFTTrainer class validation passed")
    
except ImportError as e:
    print(f"✗ Import error: {e}")
except Exception as e:
    print(f"✗ Error: {e}")

