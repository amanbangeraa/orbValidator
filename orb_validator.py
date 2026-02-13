import cv2
import numpy as np
from typing import Tuple, Optional, List
from config import Config

class ORBValidator:
    """
    ORB-based image validation system for IRoC-U 2026 compliance.
    
    Uses ORB feature detection, KNN matching with Lowe's ratio test,
    and homography-based geometric verification for robust validation.
    """
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        
        # Initialize ORB detector
        self.orb = cv2.ORB_create(nfeatures=self.config.ORB_MAX_FEATURES)
        
        # Initialize BFMatcher with Hamming distance for ORB descriptors
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    
    def extract_features(self, image: np.ndarray) -> Tuple[Optional[List], Optional[np.ndarray]]:
        """
        Extract ORB features from an image.
        
        Args:
            image: Input image (grayscale or color)
            
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
            
        # Detect ORB keypoints and compute descriptors
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        if descriptors is None or len(keypoints) == 0:
            return None, None
            
        return keypoints, descriptors
    
    def match_features(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """
        Perform KNN feature matching with Lowe's ratio test.
        
        Args:
            desc1: Descriptors from first image
            desc2: Descriptors from second image
            
        Returns:
            List of good matches after ratio test filtering
        """
        if desc1 is None or desc2 is None:
            return []
            
        # Perform KNN matching
        matches = self.matcher.knnMatch(desc1, desc2, k=self.config.KNN_K)
        
        # Apply Lowe's ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < self.config.LOWE_RATIO_THRESHOLD * n.distance:
                    good_matches.append(m)
                    
        return good_matches
    
    def verify_geometry(self, kp1: List, kp2: List, matches: List) -> Tuple[int, float]:
        """
        Perform geometric verification using homography and RANSAC.
        
        Args:
            kp1: Keypoints from first image
            kp2: Keypoints from second image
            matches: Good matches from feature matching
            
        Returns:
            Tuple of (inlier_count, inlier_ratio)
        """
        if len(matches) < self.config.MIN_HOMOGRAPHY_POINTS:
            return 0, 0.0
            
        # Extract matched point coordinates
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        try:
            # Find homography using RANSAC
            homography, mask = cv2.findHomography(
                src_pts, dst_pts,
                cv2.RANSAC,
                self.config.RANSAC_REPROJ_THRESHOLD,
                maxIters=self.config.RANSAC_MAX_ITERS,
                confidence=self.config.RANSAC_CONFIDENCE
            )
            
            if homography is None or mask is None:
                return 0, 0.0
                
            # Count inliers
            inlier_count = np.sum(mask)
            inlier_ratio = inlier_count / len(matches) if len(matches) > 0 else 0.0
            
            return int(inlier_count), float(inlier_ratio)
            
        except cv2.error:
            return 0, 0.0
    
    def validate(self, seed_img: np.ndarray, query_img: np.ndarray) -> Tuple[bool, float]:
        """
        Validate query image against seed image using ORB features.
        
        Args:
            seed_img: Reference seed image
            query_img: Query image to validate
            
        Returns:
            Tuple of (is_valid, confidence_score)
        """
        # Extract features from both images
        seed_kp, seed_desc = self.extract_features(seed_img)
        query_kp, query_desc = self.extract_features(query_img)
        
        # Check if feature extraction succeeded
        if seed_desc is None or query_desc is None:
            if self.config.VERBOSE:
                print("  ❌ Feature extraction failed")
            return False, 0.0
        
        # Perform feature matching
        good_matches = self.match_features(seed_desc, query_desc)
        num_good_matches = len(good_matches)
        
        if self.config.VERBOSE:
            print(f"  📊 Good matches: {num_good_matches}")
        
        # Check minimum good matches threshold
        if num_good_matches < self.config.MIN_GOOD_MATCHES:
            if self.config.VERBOSE:
                print(f"  ❌ Insufficient good matches ({num_good_matches} < {self.config.MIN_GOOD_MATCHES})")
            return False, 0.0
        
        # Perform geometric verification
        inlier_count, inlier_ratio = self.verify_geometry(seed_kp, query_kp, good_matches)
        
        if self.config.VERBOSE:
            print(f"  📊 Inliers: {inlier_count}/{num_good_matches} (ratio: {inlier_ratio:.3f})")
        
        # Check geometric verification thresholds
        if inlier_ratio < self.config.MIN_INLIER_RATIO:
            if self.config.VERBOSE:
                print(f"  ❌ Low inlier ratio ({inlier_ratio:.3f} < {self.config.MIN_INLIER_RATIO})")
            return False, inlier_ratio
        
        # Calculate confidence score based on inlier ratio and match count
        confidence_score = min(1.0, inlier_ratio * (num_good_matches / self.config.MIN_GOOD_MATCHES))
        
        if self.config.VERBOSE:
            print(f"  ✅ Validation passed (confidence: {confidence_score:.3f})")
        
        return True, confidence_score
    
    def validate_against_seeds(self, seed_images: List[np.ndarray], query_img: np.ndarray) -> Tuple[bool, float, int]:
        """
        Validate query image against multiple seed images and select best match.
        
        Args:
            seed_images: List of seed images
            query_img: Query image to validate
            
        Returns:
            Tuple of (is_valid, best_confidence, best_seed_index)
        """
        best_confidence = 0.0
        best_seed_idx = -1
        best_is_valid = False
        
        for i, seed_img in enumerate(seed_images):
            if self.config.VERBOSE:
                print(f"  🔍 Testing against seed {i+1}/{len(seed_images)}")
            
            is_valid, confidence = self.validate(seed_img, query_img)
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_seed_idx = i
                best_is_valid = is_valid
        
        return best_is_valid, best_confidence, best_seed_idx