#!/usr/bin/env python3
"""
Test script to check GEIA import
"""

import os
import sys

def test_geia_import():
    """Test if GEIA data_process can be imported"""
    print("Testing GEIA import...")
    
    # Try multiple possible paths
    possible_paths = [
        os.path.join(os.getcwd(), 'GEIA'),
        os.path.join(os.getcwd(), '..', 'GEIA'),
        'GEIA',
        '../GEIA'
    ]
    
    for geia_path in possible_paths:
        print(f"Trying path: {geia_path}")
        if os.path.exists(geia_path):
            print(f"✅ Path exists: {geia_path}")
            print(f"Files in directory: {os.listdir(geia_path)}")
            
            # Add to path and try import
            sys.path.insert(0, geia_path)
            try:
                from data_process import get_sent_list
                print(f"✅ Successfully imported get_sent_list from {geia_path}")
                
                # Test the function
                config = {'dataset': 'sst2', 'data_type': 'train'}
                sentences = get_sent_list(config)
                print(f"✅ Function works! Got {len(sentences)} sentences")
                print(f"Sample: {sentences[:2]}")
                return True
                
            except ImportError as e:
                print(f"❌ Import failed: {e}")
                sys.path.remove(geia_path)
                continue
            except Exception as e:
                print(f"❌ Function failed: {e}")
                sys.path.remove(geia_path)
                continue
        else:
            print(f"❌ Path does not exist: {geia_path}")
    
    return False

if __name__ == "__main__":
    success = test_geia_import()
    if success:
        print("\n🎉 GEIA import test passed!")
    else:
        print("\n❌ GEIA import test failed!") 