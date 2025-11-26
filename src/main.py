import sys
import argparse

def main():
    """
    Main entry point for the Modular Monolith application.
    """
    parser = argparse.ArgumentParser(description="CerebroVial - Modular Monolith Entry Point")
    parser.add_argument('module', choices=['vision', 'prediction', 'control'], help="Module to run")
    
    args = parser.parse_args()
    
    print(f"Starting module: {args.module}")
    
    if args.module == 'vision':
        print("Initializing Computer Vision Module...")
        # TODO: Import and run vision pipeline
    elif args.module == 'prediction':
        print("Initializing Congestion Prediction Module...")
        # TODO: Import and run prediction pipeline
    elif args.module == 'control':
        print("Initializing Control Module...")
        # TODO: Import and run control service

if __name__ == "__main__":
    main()
