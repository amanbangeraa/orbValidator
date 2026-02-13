#!/usr/bin/env python3
"""
ORB-based image validation system for UAV seed matching.

This module implements the core ORBValidator class that uses OpenCV's ORB
feature detector and descriptor matcher to validate query images against
reference seed images.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional
from config import Config


class ORBValidator:
    """
    ORB-based image validator for matching query images against seed references.
    
    Uses ORB (Oriented FAST and Rotated BRIEF) features with FLANN-based matching
    and homography verification for robust image validation.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the ORB validator with configuration parameters.
        
        Args:
            config: Configuration object containing validation thresholds
        """
        self.config = config
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=config.ORB_MAX_FEATURES)
        
        # Initialize FLANN matcher
        FLANN_INDEX_LSH = 6
        index_params = dict(
            algorithm=FLANN_INDEX_LSH,
            table_number=6,
            key_size=12,
            multi_probe_level=1
        )
        search_params = dict(checks=50)
        self.flann = cv2.FlannBasedMatcher(index_params, search_params)
        
        # Verbose output flag
        self.verbose = getattr(config, 'VERBOSE', False)
    
    def _extract_features(self, image: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Extract ORB keypoints and descriptors from an image.
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            Tuple of (keypoints, descriptors) or (None, None) if extraction fails
        """
        if image is None:
            return None, None
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
            
        # Detect keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if self.verbose:
            print(f"    Extracted {len(keypoints) if keypoints else 0} ORB features")
            
        return keypoints, descriptors
    
    def _match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List[cv2.DMatch]:
        """
        Match features between two descriptor sets using FLANN matcher.
        
        Args:
            desc1: First set of descriptors
            desc2: Second set of descriptors
            
        Returns:
            List of good matches after Lowe's ratio test
        """
        if desc1 is None or desc2 is None or len(desc1) < 2 or len(desc2) < 2:
            return []
            
        try:
            # FLANN matching
            matches = self.flann.knnMatch(desc1, desc2, k=2)
            
            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.config.LOWE_RATIO_THRESHOLD * n.distance:
                        good_matches.append(m)
                        
            if self.verbose:
                print(f"    Found {len(good_matches)} good matches (after Lowe's ratio test)")
                
            return good_matches
            
        except cv2.error as e:
            if self.verbose:
                print(f"    Matching failed: {e}")
            return []
    
    def _verify_homography(self, kp1: List[cv2.KeyPoint], kp2: List[cv2.KeyPoint], 
                          matches: List[cv2.DMatch]) -> Tuple[bool, float]:
        """
        Verify spatial consistency using homography estimation.
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: Good matches between the images
            
        Returns:
            Tuple of (is_valid, inlier_ratio)
        """
        if len(matches) < self.config.MIN_GOOD_MATCHES:
            if self.verbose:
                print(f"    Insufficient matches: {len(matches)} < {self.config.MIN_GOOD_MATCHES}")
            return False, 0.0
            
        # Extract matched point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        try:
            # Find homography using RANSAC
            homography, mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC,
                self.config.RANSAC_REPROJ_THRESHOLD
            )
            
            if homography is None:
                if self.verbose:
                    print("    Homography estimation failed")
                return False, 0.0
                
            # Calculate inlier ratio
            inliers = np.sum(mask)
            inlier_ratio = inliers / len(matches)
            
            if self.verbose:
                print(f"    Inliers: {inliers}/{len(matches)} ({inlier_ratio:.3f})")
                
            is_valid = inlier_ratio >= self.config.MIN_INLIER_RATIO
            return is_valid, inlier_ratio
            
        except cv2.error as e:
            if self.verbose:
                print(f"    Homography verification failed: {e}")
            return False, 0.0
    
    def validate(self, seed_image: np.ndarray, query_image: np.ndarray) -> Tuple[bool, float]:
        """
        Validate a query image against a single seed image.
        
        Args:
            seed_image: Reference seed image
            query_image: Query image to validate
            
        Returns:
            Tuple of (is_valid, confidence_score)
        """
        if self.verbose:
            print("  🔍 Validating image pair...")
            
        # Extract features from both images
        seed_kp, seed_desc = self._extract_features(seed_image)
        query_kp, query_desc = self._extract_features(query_image)
        
        if seed_desc is None or query_desc is None:
            if self.verbose:
                print("    ❌ Feature extraction failed")
            return False, 0.0
            
        # Match features
        matches = self._match_features(seed_desc, query_desc)
        
        if not matches:
            if self.verbose:
                print("    ❌ No matches found")
            return False, 0.0
            
        # Verify spatial consistency
        is_valid, confidence = self._verify_homography(seed_kp, query_kp, matches)
        
        if self.verbose:
            status = "✅ VALID" if is_valid else "❌ INVALID"
            print(f"    {status} (confidence: {confidence:.3f})")
            
        return is_valid, confidence
    
    def validate_against_seeds(self, seed_images: List[np.ndarray], 
                             query_image: np.ndarray) -> Tuple[bool, float, int, List[Tuple[int, float, bool]]]:
        """
        Validate a query image against multiple seed images.
        
        Args:
            seed_images: List of reference seed images
            query_image: Query image to validate
            
        Returns:
            Tuple of (is_valid, best_confidence, best_seed_idx, seed_results)
            where seed_results is a list of (idx, score, passed) for each seed
        """
        if not seed_images:
            return False, 0.0, -1, []
            
        seed_results = []
        best_confidence = 0.0
        best_seed_idx = -1
        
        for idx, seed_image in enumerate(seed_images):
            if self.verbose:
                print(f"  Testing against seed {idx + 1}/{len(seed_images)}")
                
            is_valid, confidence = self.validate(seed_image, query_image)
            seed_results.append((idx, confidence, is_valid))
            
            if is_valid and confidence > best_confidence:
                best_confidence = confidence
                best_seed_idx = idx
        
        # Overall validation passes if any seed matches
        overall_valid = best_seed_idx >= 0
        
        if self.verbose:
            valid_count = sum(1 for _, _, passed in seed_results if passed)
            print(f"  📊 Results: {valid_count}/{len(seed_images)} seeds matched")
            if overall_valid:
                print(f"  🎯 Best match: seed {best_seed_idx} (confidence: {best_confidence:.3f})")
        
        return overall_valid, best_confidence, best_seed_idx, seed_results