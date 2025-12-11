import cv2
import numpy as np

class StopSignDetector:
  
    def __init__(self, reference_image_path, min_matches=10, ransac_threshold=5.0):
        
        #Initialize the detector with a reference stop sign image.
        
        #Args: reference_image_path, min_matches, ransac_threshold
        
        self.min_matches = min_matches
        self.ransac_threshold = ransac_threshold
        
        #load reference image
        self.ref_img = cv2.imread(reference_image_path, 0)
        if self.ref_img is None:
            raise ValueError(f"Could not load reference image: {reference_image_path}")
        
        #Initialize ORB detector with more features for better matching
        self.orb = cv2.ORB_create(nfeatures=2000)
        
        #find keypoints and compute descriptors for reference image
        self.ref_keypoints, self.ref_descriptors = self.orb.detectAndCompute(self.ref_img, None)
        
        #Initialize Brute Force Matcher with hamming distance
        self.bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        print(f"Reference image loaded: {self.ref_img.shape}")
        print(f"Reference keypoints detected: {len(self.ref_keypoints)}")
    
    def detect_in_frame(self, frame):
       
        #Detect stop sign in a single frame.
        
        #Args:frame: Input frame
            
        #Returns: detected: Boolean if stop sign was detected, corners: Corner points of detected sign,  num_matches: Num good matches found
        
        #Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        #Detect keypoints and descriptors in frame
        keypoints, descriptors = self.orb.detectAndCompute(gray, None)
        
        #if we have descriptors
        if descriptors is None or len(keypoints) < self.min_matches:
            return False, None, 0
        
        matches = self.bf.match(self.ref_descriptors, descriptors) #match descriptors between reference and frame

        
        
        matches = sorted(matches, key=lambda x: x.distance)# Sort matches by distance 
        
        
        if len(matches) < self.min_matches: #if enough matches
            return False, None, len(matches)
        
        #extract matched keypoint locations
        src_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        
        #Use RANSAC to find homography and filter outliers
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, self.ransac_threshold)
        
        #Count inliers - matches that fit the homography
        if mask is not None:
            inliers = mask.ravel().tolist().count(1)
        else:
            return False, None, len(matches)
        
        #if we have enough inliers for detection
        if inliers < self.min_matches:
            return False, None, inliers
        
        #get corner points of reference image
        h, w = self.ref_img.shape
        ref_corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        
        detected_corners = cv2.perspectiveTransform(ref_corners, H) #Transform corners using homography to get detected region

        



        #-pt3 aspect ratio check
        pts = detected_corners.reshape(4, 2)
        width = np.linalg.norm(pts[1] - pts[0])
        height = np.linalg.norm(pts[2] - pts[1])
        aspect_ratio = width / height
        #print("aspect ratio" + str(aspect_ratio))
        #if not (0.3 < aspect_ratio < 3.0):
        #    return False, None, inliers  # Reject bad aspect ratio
        
        #relative area check
        area = cv2.contourArea(detected_corners)
        frame_area = frame.shape[0] * frame.shape[1]
        relative_area = area / frame_area
        #print("rel area" + str(relative_area))
        #if not (0.0005 < relative_area < 0.9):
         #   return False, None, inliers  # Reject unrealistic size
    
        return True, detected_corners, inliers
    
    def draw_detection(self, frame, corners, num_matches):
        
        #Draw bounding box and info on frame.
       
        frame = cv2.polylines(frame, [np.int32(corners)], True, (0, 255, 0), 3)
        
        #add text info
        cv2.putText(frame, f"Stop Sign Detected", (10, 250),
                   cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        cv2.putText(frame, f"Matches: {num_matches}", (10, 350),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        return frame
    
    def process_video(self, video_path, output_path=None, display=True):
        
        #open the video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing video: {video_path}")
        print(f"Resolution: {width}x{height}, FPS: {fps}, Total frames: {total_frames}")
        
        #setup video writer if output path provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        detections = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            #Detect stop sign
            detected, corners, num_matches = self.detect_in_frame(frame)
            
            if detected:
                detections += 1
                frame = self.draw_detection(frame, corners, num_matches)
            else:
                #write "searching" message
                cv2.putText(frame, "Searching...", (10, 300),
                           cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 0, 255), 5)
            
            #Write frame to output
            if writer:
                writer.write(frame)
            
            #Display frame
            if display:
                cv2.imshow('Stop Sign Detection', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\nProcessing stopped by user")
                    break
            
            if frame_count % 30 == 0:
                print(f"Progress: {frame_count}/{total_frames} frames, "
                      f"Detections: {detections}")
        
        
        cap.release()
        if writer:
            writer.release()
        if display:
            cv2.destroyAllWindows()
        
        print(f"\nProcessing complete!")
        print(f"Total frames: {frame_count}")
        print(f"Frames with detection: {detections} ({100*detections/frame_count:.1f}%)")
        if output_path:
            print(f"Output saved to: {output_path}")


if __name__ == "__main__":
  
    
    
    REFERENCE_IMAGE = "C:/Users/abhis/Documents/5550/stopsign/stop3.jpg"  
    VIDEO_PATH = "C:/Users/abhis/Downloads/drive-download-20251207T210936Z-1-001/svid4.mov"                
    OUTPUT_PATH = "C:/Users/abhis/Documents/5550/stopsign/output_detected4.mp4"          
    try:
        #initialize detector
        detector = StopSignDetector(
            reference_image_path=REFERENCE_IMAGE,
            min_matches=10,           #min matches for detect
            ransac_threshold=3.0    #RANSAC error threshold
        )
        
        detector.process_video(
            video_path=VIDEO_PATH,
            output_path=OUTPUT_PATH,
            display=True              
        )
    except Exception as e:
        print(f"Error: {e}")
