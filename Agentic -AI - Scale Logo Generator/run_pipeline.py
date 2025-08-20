#!/usr/bin/env python3
"""
Simple launcher for the Logo Generation Pipeline

This script provides an easy way to run the pipeline with examples or custom input.
"""

import sys
import os
from logo_pipeline import LogoGenerationPipeline
from examples import get_all_examples, print_example

def main():
    """Main launcher function."""
    print("🎨 Logo Generation Pipeline Launcher")
    print("=" * 50)
    
    # Check if API key is configured
    try:
        from api_config import OPENAI_API_KEY
        if not OPENAI_API_KEY or OPENAI_API_KEY == "your_openai_api_key_here":
            print("❌ Error: Please set your OpenAI API key in api_config.py")
            print("   Get your API key from: https://platform.openai.com/api-keys")
            return
        print("✅ API key configured")
    except ImportError:
        print("❌ Error: api_config.py not found")
        return
    
    # Show available examples
    examples = get_all_examples()
    print(f"\n📚 Available examples ({len(examples)}):")
    for i, name in enumerate(examples.keys(), 1):
        print(f"  {i}. {name}")
    
    print("\n" + "=" * 50)
    print("Choose an option:")
    print("1. Use an example")
    print("2. Custom input")
    print("3. Exit")
    
    while True:
        choice = input("\nEnter your choice (1-3): ").strip()
        
        if choice == "1":
            use_example(examples)
            break
        elif choice == "2":
            custom_input()
            break
        elif choice == "3":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice. Please enter 1, 2, or 3.")

def use_example(examples):
    """Use a pre-defined example."""
    print("\n" + "=" * 50)
    print("Available examples:")
    for i, name in enumerate(examples.keys(), 1):
        print(f"  {i}. {name}")
    
    while True:
        try:
            choice = int(input(f"\nSelect example (1-{len(examples)}): "))
            if 1 <= choice <= len(examples):
                example_names = list(examples.keys())
                selected_name = example_names[choice - 1]
                example = examples[selected_name]
                
                print(f"\n🎯 Selected: {selected_name}")
                print_example(selected_name)
                
                # Confirm and run
                confirm = input("\nRun pipeline with this example? (y/n): ").strip().lower()
                if confirm in ['y', 'yes']:
                    run_pipeline(example["club_description"], example["personal_vision"])
                break
            else:
                print(f"❌ Please enter a number between 1 and {len(examples)}")
        except ValueError:
            print("❌ Please enter a valid number")

def custom_input():
    """Get custom club description and vision."""
    print("\n" + "=" * 50)
    print("Enter your custom club information:")
    
    club_description = input("\nClub Description:\n").strip()
    if not club_description:
        print("❌ Club description is required")
        return
    
    personal_vision = input("\nPersonal Vision (optional):\n").strip()
    
    print("\n" + "=" * 50)
    print("Club Description:")
    print(club_description)
    if personal_vision:
        print("\nPersonal Vision:")
        print(personal_vision)
    
    # Confirm and run
    confirm = input("\nRun pipeline with this input? (y/n): ").strip().lower()
    if confirm in ['y', 'yes']:
        run_pipeline(club_description, personal_vision)

def run_pipeline(club_description: str, personal_vision: str = ""):
    """Run the logo generation pipeline."""
    print("\n🚀 Starting Logo Generation Pipeline...")
    print("=" * 50)
    
    try:
        # Initialize pipeline
        pipeline = LogoGenerationPipeline()
        
        # Run the pipeline
        result = pipeline.run_pipeline(club_description, personal_vision)
        
        if result:
            print("\n✅ Pipeline completed successfully!")
            print(f"📁 Results saved in: {pipeline.storage.project_dir}")
        else:
            print("\n❌ Pipeline failed. Check the error messages above.")
            
    except Exception as e:
        print(f"\n❌ Error running pipeline: {e}")
        print("Please check your configuration and try again.")

if __name__ == "__main__":
    main()
