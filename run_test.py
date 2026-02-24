#!/usr/bin/env python3
"""
IRoC-U 2026 ORB Seed Validation Test Runner with Ambiguity Rejection

This script demonstrates the complete validation pipeline with enhanced ambiguity handling:
1. Load all seed images from data/seeds/
2. Load all query images from data/queries/
3. Match each query against all seeds with ambiguity rejection
4. Output validation results in required format
"""

import cv2
import os
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np

from orb_validator_complete import ORBValidator
from config import Config

def load_images_from_directory(directory: str) -> Tuple[List[np.ndarray], List[str]]:
    """
    Load all images from a directory.
    
    Args:
        directory: Path to directory containing images
        
    Returns:
        Tuple of (images_list, filenames_list)
    """
    images = []
    filenames = []
    
    if not os.path.exists(directory):
        print(f"⚠️  Directory not found: {directory}")
        return images, filenames
    
    # Supported image extensions
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    for filename in sorted(os.listdir(directory)):
        if Path(filename).suffix.lower() in extensions:
            filepath = os.path.join(directory, filename)
            img = cv2.imread(filepath)
            
            if img is not None:
                images.append(img)
                filenames.append(filename)
                print(f"📁 Loaded: {filename}")
            else:
                print(f"⚠️  Failed to load: {filename}")
    
    return images, filenames

def main():
    """
    Main validation test runner with ambiguity rejection.
    """
    print("🚁 IRoC-U 2026 ORB Seed Validation System (Enhanced)")
    print("=" * 50)
    
    # Initialize validator
    config = Config()
    validator = ORBValidator(config)
    
    # Load seed images
    print("\n📂 Loading seed images...")
    seed_images, seed_filenames = load_images_from_directory("data/seeds")
    
    if not seed_images:
        print("❌ No seed images found!")
        return 1
    
    print(f"✅ Loaded {len(seed_images)} seed images")
    
    # Load query images
    print("\n📂 Loading query images...")
    queries, query_filenames = load_images_from_directory("data/queries")
    
    if not queries:
        print("❌ No query images found!")
        return 1
    
    print(f"✅ Loaded {len(queries)} query images")
    
    # Validation results
    print("\n🔍 Running validation tests...")
    print("=" * 70)
    print(f"{'QUERY IMAGE':<20} → {'BEST SEED':<15} | {'SCORE':<8} | {'RESULT':<20}")
    print("-" * 70)
    
    total_valid = 0
    AMBIGUITY_MARGIN = 0.01  # Very low threshold - accept best match even if close

    for query_img, query_filename in zip(queries, query_filenames):
        print(f"\n🔍 Processing: {query_filename}")

        is_valid, confidence, best_seed_idx, seed_results = \
            validator.validate_against_seeds(seed_images, query_img)

        # Collect all passing seeds
        valid_candidates = [
            (idx, score)
            for idx, score, passed in seed_results
            if passed
        ]

        result_status = "INVALID"
        best_seed_name = "NONE"

        if is_valid and len(valid_candidates) == 1:
            result_status = "VALID"
            best_seed_name = seed_filenames[best_seed_idx]
            total_valid += 1

        elif is_valid and len(valid_candidates) > 1:
            # Ambiguity rejection
            valid_candidates.sort(key=lambda x: x[1], reverse=True)
            best_score = valid_candidates[0][1]
            second_best_score = valid_candidates[1][1]

            if best_score - second_best_score >= AMBIGUITY_MARGIN:
                result_status = "VALID"
                best_seed_idx = valid_candidates[0][0]
                best_seed_name = seed_filenames[best_seed_idx]
                confidence = best_score
                total_valid += 1
            else:
                result_status = "INVALID (AMBIGUOUS)"
                confidence = best_score

        print(
            f"{query_filename:<20} → "
            f"{best_seed_name:<15} | "
            f"{confidence:<8.3f} | "
            f"{result_status}"
        )
    
    # Summary
    print("\n" + "=" * 70)
    print(f"📊 VALIDATION SUMMARY")
    print(f"   Total queries: {len(queries)}")
    print(f"   Valid matches: {total_valid}")
    print(f"   Invalid: {len(queries) - total_valid}")
    print(f"   Success rate: {(total_valid/len(queries))*100:.1f}%")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())

