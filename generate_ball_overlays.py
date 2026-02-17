
import cv2
import csv
import glob
import os
import sys

def process_video(video_path, csv_path, output_path):
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return

    # Video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps == 0:
        fps = 30  # Default if reading fails

    # Calculate output path
    print(f"Processing: {os.path.basename(video_path)} -> {os.path.basename(output_path)}")

    # Prepare VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not create output video writer for {output_path}")
        cap.release()
        return

    # Load ball annotations
    ball_data = {}
    try:
        with open(csv_path, 'r', newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    # CSV format: frame,ball_x,ball_y
                    # frame field format: 'frame_XXX'
                    frame_id_str = row['frame']
                    if '_' in frame_id_str:
                        frame_idx = int(frame_id_str.split('_')[1])
                    else:
                        frame_idx = int(frame_id_str) # direct integer fallback

                    x = float(row['ball_x'])
                    y = float(row['ball_y'])
                    
                    # Check for valid coordinates (not NaN)
                    if x == x and y == y: # float('nan') != float('nan')
                        ball_data[frame_idx] = (int(x), int(y))
                except (ValueError, KeyError, IndexError):
                    continue
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")
        cap.release()
        out.release()
        return

    # Process frames
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Check if we have ball data for this frame
        if frame_idx in ball_data:
            center = ball_data[frame_idx]
            # Draw point: Size 3 (radius), filled
            cv2.circle(frame, center, 3, (0, 0, 255), -1)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    print(f"Finished writing {output_path}")


def main():
    # Configuration
    data_dir = "data/tenis-backview"
    output_dir = "output_videos"
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all mp4 files
    video_files = glob.glob(os.path.join(data_dir, "*.mp4"))
    
    if not video_files:
        print(f"No .mp4 files found in {data_dir}")
        return

    print(f"Found {len(video_files)} videos to process.")

    for video_path in video_files:
        filename = os.path.basename(video_path)
        basename, ext = os.path.splitext(filename)
        
        # Determine CSV path
        # Pattern: videoX.mp4 -> videoX_ball.csv
        csv_filename = f"{basename}_ball.csv"
        csv_path = os.path.join(data_dir, csv_filename)
        
        if not os.path.exists(csv_path):
            print(f"Skipping {filename}: Annotation file {csv_filename} not found.")
            continue
            
        output_filename = f"{basename}_ball.mp4"
        output_path = os.path.join(output_dir, output_filename)
        
        process_video(video_path, csv_path, output_path)

if __name__ == "__main__":
    main()
