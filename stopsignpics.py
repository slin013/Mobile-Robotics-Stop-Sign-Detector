import cv2
import numpy as np

"""
Minimal ORB + RANSAC stop sign detector for static images.
Based on: https://www.geeksforgeeks.org/python/step-by-step-guide-to-using-ransac-in-opencv-using-python/
"""

# ============ CONFIGURATION ============
REFERENCE_IMAGE = "C:/Users/abhis/Documents/5550/stopsign/stop3.jpg"  # Your reference stop sign
TEST_IMAGE = "C:/Users/abhis/Documents/5550/stopsign/stop6.jpg"                # Image to search for stop sign
MIN_MATCHES = 10                              # Minimum matchs for detection
RANSAC_THRESHOLD = 5.0                        # RANSAC error threshold (pixels)

# ============ LOAD IMAGES ============
ref_img = cv2.imread(REFERENCE_IMAGE, 0)  # Load reference as grayscale
test_img = cv2.imread(TEST_IMAGE, 0)      # Load test image as grayscale
test_img_color = cv2.imread(TEST_IMAGE)   # Load test image in color for display

if ref_img is None or test_img is None:
    raise ValueError("Could not load images. Check file paths!")

print(f"Reference image: {ref_img.shape}")
print(f"Test image: {test_img.shape}")

# ============ DETECT KEYPOINTS ============
orb = cv2.ORB_create(nfeatures=2000)

# Detect and compute for reference image
kp1, des1 = orb.detectAndCompute(ref_img, None)
print(f"Reference keypoints: {len(kp1)}")

# Detect and compute for test image
kp2, des2 = orb.detectAndCompute(test_img, None)
print(f"Test image keypoints: {len(kp2)}")

# ============ MATCH DESCRIPTORS ============
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

print(f"Total matches found: {len(matches)}")

# ============ APPLY RANSAC ============
if len(matches) >= MIN_MATCHES:
    # Extract matched keypoint locations
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    # Find homography using RANSAC
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, RANSAC_THRESHOLD)
    
    # Count inliers
    inliers = mask.ravel().tolist().count(1)
    print(f"Inliers (good matches): {inliers}")
    
    if inliers >= MIN_MATCHES:
        print("\n✓ STOP SIGN DETECTED!")
        
        # Get corners of reference image
        h, w = ref_img.shape
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        # Transform corners to test image coordinates
        detected_corners = cv2.perspectiveTransform(corners, H)
        
        # Draw bounding box on test image
        test_img_color = cv2.polylines(test_img_color, [np.int32(detected_corners)], 
                                       True, (0, 255, 0), 3)
        
        # Add text
        cv2.putText(test_img_color, "Stop Sign Detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(test_img_color, f"Matches: {inliers}", (10, 70),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # ============ DISPLAY RESULTS ============
        # Show matched keypoints
        match_img = cv2.drawMatches(ref_img, kp1, test_img, kp2, 
                                    matches[:50], None, 
                                    flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        
        # Display images
        cv2.imshow('Matched Keypoints', match_img)
        cv2.imshow('Detection Result', test_img_color)
        
        # Save results
        cv2.imwrite('output_matches.jpg', match_img)
        cv2.imwrite('output_detection.jpg', test_img_color)
        print("\nResults saved:")
        print("- output_matches.jpg (keypoint matches)")
        print("- output_detection.jpg (detected stop sign with box)")
        
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("\n✗ Not enough inliers for reliable detection")
else:
    print("\n✗ Not enough matches found")

print("\nHomography Matrix (H):")
print(H)