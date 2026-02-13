# orbValidator
# ORB Seed Validation System - IRoC-U 2026

A rule-compliant validation system for UAV-captured images using ORB feature matching, designed specifically for the IRoC-U 2026 "Validation Task" requirements.

## Overview

This system validates UAV-captured query images against reference seed images using computer vision techniques that are deterministic, explainable, and require no training or external services.

## How ORB is Used

**ORB (Oriented FAST and Rotated BRIEF)** is chosen for its key advantages:

- **Rotation Invariant**: Critical for UAV images captured at different orientations
- **Scale Invariant**: Handles altitude variations in UAV imagery  
- **Fast Computation**: Real-time capable for competition requirements
- **Binary Descriptors**: Efficient Hamming distance matching
- **Deterministic**: Same image always produces same features

The system extracts up to 5,000 ORB keypoints and descriptors from each image, providing robust feature representation even under challenging conditions.

## Feature Matching Pipeline

1. **ORB Feature Extraction**: Detect keypoints and compute binary descriptors
2. **KNN Matching**: Find 2 nearest neighbors for each descriptor using BFMatcher with Hamming distance
3. **Lowe's Ratio Test**: Filter matches where `distance(best) < 0.75 × distance(second_best)`
4. **Geometric Verification**: Use RANSAC-based homography to eliminate outliers

## Why Homography Verification is Required

Homography verification is essential for IRoC-U compliance because:

- **Geometric Consistency**: Ensures matched features follow realistic spatial transformations
- **Outlier Rejection**: RANSAC eliminates false positive matches from similar textures
- **Perspective Correction**: Accounts for UAV viewing angle differences
- **Robust Validation**: Requires spatially coherent feature matches, not just descriptor similarity

Without homography verification, the system could incorrectly validate images with similar textures but different scenes.

## Validation Thresholds

The system uses multiple configurable thresholds for robust validation:

### Primary Thresholds
- **Minimum Good Matches**: 20 (ensures sufficient feature overlap)
- **Minimum Inlier Ratio**: 30% (geometric consistency requirement)
- **Lowe Ratio**: 0.75 (feature distinctiveness filter)

### RANSAC Parameters
- **Reprojection Threshold**: 5.0 pixels (geometric tolerance)
- **Max Iterations**: 2000 (thorough outlier detection)
- **Confidence**: 99% (high reliability requirement)

### Decision Logic
A query image is marked **VALID** only if:
1. ✅ ORB features extracted successfully from both images
2. ✅ At least 20 good matches found after Lowe's ratio test
3. ✅ Homography computed successfully with RANSAC
4. ✅ At least 30% of good matches are geometric inliers

## IRoC-U "Automatic Image-Matching Validation" Compliance

This system satisfies IRoC-U requirements through:

### ✅ **No Deep Learning**: Uses classical computer vision (ORB + RANSAC)
### ✅ **No Training Required**: Deterministic algorithms with no learning phase
### ✅ **Explainable Logic**: Every decision based on measurable geometric criteria
### ✅ **Deterministic Results**: Same inputs always produce same outputs
### ✅ **Real-time Capable**: Fast ORB features and efficient matching
### ✅ **Multi-seed Support**: Validates against multiple reference images
### ✅ **Configurable Thresholds**: Tunable parameters for different scenarios

## Installation & Usage

### Requirements
```bash
pip install opencv-python numpy
```

### Directory Structure
```
orb_seed_validation/
├── data/
│   ├── seeds/          # Reference images
│   └── queries/        # Images to validate
├── config.py           # Tunable parameters
├── orb_validator.py    # Core validation logic
├── run_test.py         # Test runner
└── README.md          # This file
```

### Running the System
```bash
# Place reference images in data/seeds/
# Place test images in data/queries/
python run_test.py
```

### Expected Output Format
```
QUERY IMAGE          → BEST SEED       | SCORE    | RESULT
--------------------------------------------------------------------
drone_001.jpg        → seed_A.jpg      | 0.847    | VALID
drone_002.jpg        → seed_B.jpg      | 0.234    | INVALID
drone_003.jpg        → seed_A.jpg      | 0.692    | VALID
```

## Configuration

Edit `config.py` to adjust validation parameters:

```python
# Feature Detection
ORB_MAX_FEATURES = 5000

# Matching Thresholds  
LOWE_RATIO_THRESHOLD = 0.75
MIN_GOOD_MATCHES = 20
MIN_INLIER_RATIO = 0.3

# RANSAC Parameters
RANSAC_REPROJ_THRESHOLD = 5.0
```

## Technical Implementation Details

### Core Algorithm Flow
1. Load seed and query images
2. Extract ORB features (keypoints + descriptors)
3. Perform KNN matching with k=2
4. Apply Lowe's ratio test (distance₁ < 0.75 × distance₂)
5. Compute homography using RANSAC
6. Count geometric inliers
7. Apply validation thresholds
8. Return VALID/INVALID with confidence score

### Confidence Scoring
```python
confidence = min(1.0, inlier_ratio × (good_matches / min_threshold))
```

### Multi-Seed Validation
- Tests query against all available seed images
- Selects seed with highest confidence score
- Reports best matching seed and final validation result

## Validation Robustness

The system handles common UAV imaging challenges:
- ✅ **Rotation**: ORB rotation invariance
- ✅ **Scale Changes**: ORB scale invariance  
- ✅ **Lighting Variations**: Robust binary descriptors
- ✅ **Perspective Changes**: Homography transformation
- ✅ **Partial Occlusion**: RANSAC outlier rejection
- ✅ **Similar Textures**: Geometric verification requirement

This ensures reliable validation even under challenging field conditions typical in UAV competitions.