import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim

# Load video
def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Error opening video file: {video_path}")
    return cap

# Extract frames from video
def extract_frames(cap):
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    return frames

# Color threshold segmentation
def color_threshold_segmentation(frame, lower_bound, upper_bound):
    # Convert frame to HSV color space for better segmentation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return segmented_frame

# Sobel edge detection
def sobel_edge_detection(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sobelx = np.abs(cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5))
    sobely = np.abs(cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5))
    edge = np.sqrt(sobelx**2 + sobely**2)
    edge = np.uint8(edge / np.max(edge) * 255)
    return edge

# Segment frames using Sobel technique and color thresholding
def segment_frames(frames, lower_bound, upper_bound):
    sobel_segmented = [sobel_edge_detection(frame) for frame in frames]
    color_segmented = [color_threshold_segmentation(frame, lower_bound, upper_bound) for frame in frames]
    return sobel_segmented, color_segmented

# SSIM-based Scene Cut Detection
def detect_scene_cuts_ssim(frames, hard_cut_threshold=0.5, soft_cut_threshold=0.75):
    scene_cuts = {'hard': [], 'soft': []}
    ssim_scores = []
    
    for i in range(1, len(frames)):
        prev_frame = cv2.cvtColor(frames[i-1], cv2.COLOR_BGR2GRAY)
        curr_frame = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
        
        score, _ = ssim(prev_frame, curr_frame, full=True)
        ssim_scores.append(score)
        
        if score < hard_cut_threshold:
            scene_cuts['hard'].append(i)
        elif score < soft_cut_threshold:
            scene_cuts['soft'].append(i)
    
    return scene_cuts, ssim_scores

# Visualize SSIM results
def visualize_ssim_results(frames, scene_cuts, ssim_scores, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    
    # Timeline visualization
    timeline = np.zeros((100, len(frames), 3), dtype=np.uint8)
    
    for i in scene_cuts['hard']:
        timeline[:, i] = [255, 0, 0]  # Red for hard cuts
    for i in scene_cuts['soft']:
        timeline[:, i] = [255, 165, 0]  # Orange for soft cuts

    # Plot SSIM scores with timeline
    fig, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 10))
    
    # Plot timeline of cuts
    ax[0].imshow(timeline)
    ax[0].set_title("Timeline of Scene Cuts")
    ax[0].set_xlabel("Frames")
    ax[0].set_yticks([])
    
    # Plot SSIM scores
    ax[1].plot(range(1, len(ssim_scores)+1), ssim_scores)
    ax[1].set_title("SSIM Score Between Consecutive Frames")
    ax[1].set_xlabel("Frames")
    ax[1].set_ylabel("SSIM Score")
    
    plt.tight_layout()
    
    # Save figure
    fig.savefig(os.path.join(output_folder, 'ssim_cut_visualization.png'))
    plt.show()

# Save frames and results to disk
def save_frames(frames, segmented_frames, output_folder):
    frame_folder = os.path.join(output_folder, "frames")
    segmented_folder = os.path.join(output_folder, "segmented_frames")
    
    os.makedirs(frame_folder, exist_ok=True)
    os.makedirs(segmented_folder, exist_ok=True)
    
    for i, frame in enumerate(frames):
        frame_path = os.path.join(frame_folder, f"frame_{i:04d}.png")
        segmented_path = os.path.join(segmented_folder, f"segmented_{i:04d}.png")
        
        cv2.imwrite(frame_path, frame)
        cv2.imwrite(segmented_path, segmented_frames[i])

# Detect objects using centroid-based tracking
def detect_objects_centroid(frames):
    tracked_frames = []
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(blurred, 200, 255, cv2.THRESH_BINARY)

        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            if cv2.contourArea(contour) > 100:  # Filter small contours
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cX, cY), 5, (0, 255, 0), -1)  # Draw centroid

        tracked_frames.append(frame)

    return tracked_frames

# Save tracked frames to a new video
def save_tracked_video(tracked_frames, output_video_path, fps=30):
    if len(tracked_frames) == 0:
        print("No frames to save.")
        return

    height, width, _ = tracked_frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi files
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in tracked_frames:
        out.write(frame)

    out.release()
    print(f"Tracked video saved at: {output_video_path}")

# Calculate and visualize histograms
def calculate_histograms(frames, output_folder):
    os.makedirs(output_folder, exist_ok=True)
    colors = ('b', 'g', 'r')  # Color channels
    plt.figure(figsize=(10, 5))

    for i, color in enumerate(colors):
        hist_values = np.zeros((len(frames), 256))
        for j, frame in enumerate(frames):
            hist_values[j] = cv2.calcHist([frame], [i], None, [256], [0, 256]).flatten()

        # Plot histograms for each color channel
        plt.plot(hist_values.mean(axis=0), color=color)

    plt.title('Color Histogram for Video Frames')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend(['Blue Channel', 'Green Channel', 'Red Channel'])
    plt.savefig(os.path.join(output_folder, 'color_histogram.png'))
    plt.show()

def main():
    # Define paths
    video_path = r"C:\Users\ASUS\Downloads\nikead.mp4"  # Update to your video path
    output_folder = r"D:\\FALL SEMESTER 2024-2025\\CSE 4037 IMG & VIDEO ANALYTICS\\LABASG4\\outputf"  # Output folder path
    
    # Load video
    cap = load_video(video_path)
    frames = extract_frames(cap)
    print(f"Extracted {len(frames)} frames")
    
    # Define color thresholds for segmentation (adjust these values as needed)
    lower_bound = np.array([100, 150, 0])  # Example lower bound for HSV
    upper_bound = np.array([140, 255, 255])  # Example upper bound for HSV
    
    # Segment frames
    sobel_segmented, color_segmented = segment_frames(frames, lower_bound, upper_bound)
    
    # Detect scene cuts
    scene_cuts, ssim_scores = detect_scene_cuts_ssim(frames)
    print(f"Detected hard cuts at frames: {scene_cuts['hard']}")
    print(f"Detected soft cuts at frames: {scene_cuts['soft']}")
    
    # Visualize SSIM results
    visualize_ssim_results(frames, scene_cuts, ssim_scores, output_folder)
    
    # Save frames and segmented results
    save_frames(frames, color_segmented, output_folder)

    # Track objects using centroid
    tracked_frames = detect_objects_centroid(frames)
    
    # Save tracked video
    save_tracked_video(tracked_frames, os.path.join(output_folder, 'tracked_video.avi'))
    
    # Calculate and visualize color histograms
    calculate_histograms(frames, output_folder)

if __name__ == "__main__":
    main()


