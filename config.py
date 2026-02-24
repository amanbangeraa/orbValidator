# Configuration parameters for ORB seed validation system
# IRoC-U 2026 compliant validation parameters

class Config:
    # ORB Feature Detection Parameters
    ORB_MAX_FEATURES = 5000  # Maximum number of ORB features to detect
    
    # Feature Matching Parameters
    LOWE_RATIO_THRESHOLD = 0.75  # Lowe's ratio test threshold (lower = more strict)
    KNN_K = 2  # Number of nearest neighbors for KNN matching
    
    # Validation Thresholds
    MIN_GOOD_MATCHES = 10  # Minimum number of good matches required
    MIN_INLIER_RATIO = 0.15  # Minimum ratio of inliers to good matches (15%)
    
    # RANSAC Parameters for Homography
    RANSAC_REPROJ_THRESHOLD = 5.0  # Maximum allowed reprojection error in pixels
    RANSAC_MAX_ITERS = 2000  # Maximum RANSAC iterations
    RANSAC_CONFIDENCE = 0.99  # RANSAC confidence level
    
    # Homography Validation
    MIN_HOMOGRAPHY_POINTS = 4  # Minimum points needed for homography computation
    
    # Output Configuration
    VERBOSE = False  # Enable detailed output logging