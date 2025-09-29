#!/usr/bin/env python3
"""
Setup script for the Triton Inference Client.
"""
import os
import subprocess
import sys
from pathlib import Path


def create_directories():
    """Create necessary directories."""
    directories = ['output', 'media', 'logs']
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"Created directory: {directory}")


def install_requirements():
    """Install Python requirements."""
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("Requirements installed successfully")
    except subprocess.CalledProcessError as e:
        print(f"Failed to install requirements: {e}")
        return False
    return True


def main():
    """Main setup function."""
    print("Setting up Triton Inference Client...")
    
    # Create directories
    create_directories()
    
    # Install requirements
    if not install_requirements():
        print("Setup completed with warnings. Please check requirements installation.")
        return
    
    print("Setup completed successfully!")
    print("\nUsage:")
    print("  python main.py --help")
    print("  python main.py video media/30sec.mp4 -o output/result.mp4")


if __name__ == '__main__':
    main()
