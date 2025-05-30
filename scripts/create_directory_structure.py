#!/usr/bin/env python3

import os
import sys

def create_directories(base_dir):
    """
    Create the essential directory structure for the energy market forecasting project
    """
    directories = [
        # Main directories
        os.path.join(base_dir, "scripts"),
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "outputs"),
        
        # Data subdirectories
        os.path.join(base_dir, "data", "raw"),
        os.path.join(base_dir, "data", "processed"),
        os.path.join(base_dir, "data", "final"),
        
        # Outputs subdirectories
        os.path.join(base_dir, "outputs", "images")
    ]
    
    print("Creating directory structure...")
    
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory)
                print(f"Created: {directory}")
            except Exception as e:
                print(f"Error creating {directory}: {e}")
        else:
            print(f"Already exists: {directory}")
    
    print("\nDirectory structure creation complete.")

def validate_directories(base_dir):
    """
    Validate that all required directories exist
    """
    directories = [
        # Main directories
        os.path.join(base_dir, "scripts"),
        os.path.join(base_dir, "data"),
        os.path.join(base_dir, "models"),
        os.path.join(base_dir, "outputs"),
        
        # Data subdirectories
        os.path.join(base_dir, "data", "raw"),
        os.path.join(base_dir, "data", "processed"),
        os.path.join(base_dir, "data", "final"),
        
        # Outputs subdirectories
        os.path.join(base_dir, "outputs", "images")
    ]
    
    print("\nValidating directory structure...")
    all_valid = True
    
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ Valid: {directory}")
        else:
            print(f"❌ Missing: {directory}")
            all_valid = False
    
    if all_valid:
        print("\nAll directories exist and are valid.")
    else:
        print("\nSome directories are missing. Please run the script with the create option.")
    
    return all_valid

if __name__ == "__main__":
    # Get current directory as base directory
    base_dir = os.getcwd()
    
    # Command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "create":
            create_directories(base_dir)
        elif sys.argv[1] == "validate":
            validate_directories(base_dir)
        else:
            print("Usage: python create_directory_structure.py [create|validate]")
    else:
        # Default to validate
        if not validate_directories(base_dir):
            choice = input("\nWould you like to create the missing directories? (y/n): ")
            if choice.lower() == 'y':
                create_directories(base_dir) 