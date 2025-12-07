import cv2
import numpy as np
import os
import shutil

# =========================================================
# PREP OUTPUT FOLDER (fresh each run)
# =========================================================
output_root = "output-images"

if os.path.exists(output_root):
    shutil.rmtree(output_root)

os.makedirs(output_root)


# =========================================================
# LOAD REFERENCE IMAGE + ORB FEATURES
# =========================================================
reference_path = "reference.jpg"
ref_img = cv2.imread(reference_path)
max_dim = 500
h, w = ref_img.shape[:2]
scale = max_dim / max(h, w)
if scale < 1:
    ref_img = cv2.resize(ref_img, (int(w * scale), int(h * scale)))
ref_img = cv2.GaussianBlur(ref_img, (3,3), 0)


if ref_img is None:
    raise ValueError("Error: cannot load reference image. Check the file path!")

ref_gray = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create(nfeatures=2000)
kp1, des1 = orb.detectAndCompute(ref_gray, None)


# =========================================================
# STOP SIGN DETECTION FUNCTION
# =========================================================
def detect_stop_sign(test_img):
    max_dim = 500
    h, w = test_img.shape[:2]
    scale = max_dim / max(h, w)
    if scale < 1:
        test_img = cv2.resize(test_img, (int(w * scale), int(h * scale)))

    test_img = cv2.GaussianBlur(test_img, (3, 3), 0)

    test_gray = cv2.cvtColor(test_img, cv2.COLOR_BGR2GRAY)

    # Extract ORB features
    kp2, des2 = orb.detectAndCompute(test_gray, None)
    if des2 is None:
        output = test_img.copy()
        cv2.putText(output, "NOT DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)

        return output, False, 0

    # Match descriptors
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches[:40]

    # Extract point coordinates
    ref_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    test_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)

    # Compute homography
    H, mask = cv2.findHomography(ref_pts, test_pts, cv2.RANSAC, 5.0)
    inliers = mask.ravel().tolist().count(1) if mask is not None else 0
    print("Inliers:", inliers)

    # Detection threshold
    if H is None or inliers < 9:
        output = test_img.copy()
        cv2.putText(output, "NOT DETECTED", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,0,255), 3)
        cv2.putText(output, "INLIERS:" + str(inliers), (30, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
        return output, False, inliers

    # Draw bounding box
    h, w = ref_gray.shape
    corners = np.float32([[0,0],[w,0],[w,h],[0,h]]).reshape(-1,1,2)
    projected = cv2.perspectiveTransform(corners, H)

    output = test_img.copy()
    cv2.polylines(output, [np.int32(projected)], True, (191,64,191), 3)
    cv2.putText(output, "DETECTED", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0,255,0), 3)
    cv2.putText(output, "INLIERS:" + str(inliers), (30, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

    return output, True, inliers


# =========================================================
# PROCESS FOLDERS + COMPUTE ACCURACY
# =========================================================

folders = {
    "stop-close": "stop-close",
    "stop-angle": "stop-angle",
    "stop-far":   "stop-far",
    "no-stop":    "no-stop",
    "other-sign": "other-sign"
}

results = {}

for folder_name, folder_path in folders.items():
    print("\nProcessing folder:", folder_name)

    total_images = 0
    correct = 0

    # Create subfolder inside output-images
    save_folder = os.path.join(output_root, folder_name)
    os.makedirs(save_folder, exist_ok=True)

    for filename in os.listdir(folder_path):
        if filename.lower().endswith((".jpg", ".jpeg", ".png",".webp", ".heic")):

            total_images += 1
            img_path = os.path.join(folder_path, filename)

            test_img = cv2.imread(img_path)
            if test_img is None:
                print("Could not load:", img_path)
                continue

            # --- RUN DETECTION ---
            output_img, detected, inliers = detect_stop_sign(test_img)

            # --- SAVE TO output-images/<folder>/ ---
            name, ext = os.path.splitext(filename)
            save_path = os.path.join(save_folder, f"{name}_output{ext}")
            cv2.imwrite(save_path, output_img)

            # --- ACCURACY CALC ---
            if folder_name in ["no-stop", "other-sign"]:
                # Should NOT detect
                if not detected:
                    correct += 1
            else:
                # Should detect
                if detected:
                    correct += 1

            print(f"   {filename}: {'DETECTED' if detected else 'NOT DETECTED'} (inliers={inliers})")

    # Compute accuracy for this folder
    accuracy = (correct / total_images) * 100 if total_images > 0 else 0
    results[folder_name] = accuracy

    print(f"Accuracy for {folder_name}: {accuracy:.2f}%")


# =========================================================
# FINAL SUMMARY
# =========================================================
print("\n====== FINAL ACCURACY SUMMARY ======")
for folder_name, acc in results.items():
    print(f"{folder_name}: {acc:.2f}%")

overall_acc = sum(results.values()) / len(results)
print(f"\nOverall accuracy: {overall_acc:.2f}%")
