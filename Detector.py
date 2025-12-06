import cv2
import numpy as np

# Load images
reference_path = "stop-sign-plain.jpg"   # Reference image
test_path = "stop-sign-close.png"        # Test Image

ref_img = cv2.imread(reference_path)
test_img = cv2.imread(test_path)

if ref_img is None or test_img is None:
    raise ValueError("Error loading images. Check your file paths!")

ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

# ORB keypoints and descriptors
orb = cv2.ORB_create(nfeatures=2000)

kp1, des1 = orb.detectAndCompute(ref_gray, None)
kp2, des2 = orb.detectAndCompute(test_gray, None)

# Match descriptors using BFMatcher - Hamming distance
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)

# Sort matches by distance (smaller distance = better match)
matches = sorted(matches, key=lambda x: x.distance)

# Use the top matches for homography estimation
good_matches = matches[:40]

# Extract matched keypoints
ref_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
test_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

# Compute homography using RANSAC
H, mask = cv2.findHomography(ref_pts, test_pts, cv2.RANSAC, 5.0)

if H is not None:
    # Outline stop sign using transformed corners
    h, w = ref_gray.shape

    # Reference image corners
    corners = np.float32([[0,0], [w,0], [w,h], [0,h]]).reshape(-1,1,2)

    # Project corners into test image
    projected = cv2.perspectiveTransform(corners, H)

    # Draw polygon around detected stop sign
    output_img = test_img.copy()
    cv2.polylines(output_img, [np.int32(projected)], True, (191,64,191), 3)

    print("Stop sign detected!")

else:
    output_img = test_img
    print("Homography failed. Stop sign may not be detected.")

# Display result
cv2.imshow("Reference Image", ref_img)
cv2.imshow("Detected Stop Sign", output_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optionally save output
cv2.imwrite("detected_output.jpg", output_img)
