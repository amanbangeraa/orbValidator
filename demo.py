#!/usr/bin/env python3
"""
Simple demonstration of the ORB validation system core functionality.
"""

import cv2
import numpy as np
from orb_validator_class import ORBValidator
from config import Config

def create_test_images():
    """Create simple test images for demonstration."""
    # Create a test image with some geometric patterns
    img1 = np.zeros((400, 400, 3), dtype=np.uint8)
    
    # Add some geometric features
    cv2.rectangle(img1, (50, 50), (150, 150), (255, 255, 255), -1)
    cv2.circle(img1, (300, 100), 50, (128, 128, 128), -1)
    cv2.rectangle(img1, (200, 250), (350, 350), (200, 200, 200), -1)
    
    # Create a slightly transformed version (should match)
    img2 = cv2.resize(img1, (350, 350))  # Scale change
    img2 = cv2.rotate(img2, cv2.ROTATE_90_CLOCKWISE)  # Rotation
    
    # Create a different image (should not match)
    img3 = np.zeros((400, 400, 3), dtype=np.uint8)
    cv2.circle(img3, (200, 200), 100, (255, 255, 255), -1)
    cv2.rectangle(img3, (100, 300), (300, 350), (128, 128, 128), -1)
    
    return img1, img2, img3

def main():
    print("🔬 ORB Validation System Demo")
    print("=" * 40)
    
    # Create test images
    seed_img, matching_query, non_matching_query = create_test_images()
    
    # Initialize validator
    config = Config()
    config.VERBOSE = True  # Enable detailed output
    validator = ORBValidator(config)
    
    print("\n📊 Test 1: Matching images (transformed version)")
    print("-" * 50)
    is_valid, confidence = validator.validate(seed_img, matching_query)
    print(f"Result: {'VALID' if is_valid else 'INVALID'} (confidence: {confidence:.3f})")
    
    print("\n📊 Test 2: Non-matching images (different content)")
    print("-" * 50)
    is_valid, confidence = validator.validate(seed_img, non_matching_query)
    print(f"Result: {'VALID' if is_valid else 'INVALID'} (confidence: {confidence:.3f})")
    
    print("\n✅ Demo completed successfully!")

if __name__ == "__main__":
    main()