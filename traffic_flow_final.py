import cv2
import argparse
import json
import csv
import os
import numpy as np
from ultralytics import YOLO
import subprocess

# Function to download YouTube video if URL is provided
def download_video(source):
    if source.startswith("http"):
        print("[INFO] Downloading video from YouTube using yt-dlp...")
        output_file = os.path.join(os.getcwd(), "input_video.mp4")
        subprocess.run([
            "yt-dlp", "-f", "18", "-o", output_file, source
        ], check=True)
        return output_file
    return source

# Function to check which lane a vehicle belongs to
def get_lane(lanes, cx, cy):
    for lane in lanes['lanes']:
        pts = np.array(lane['coordinates'], np.int32)
        if cv2.pointPolygonTest(pts, (cx, cy), False) >= 0:
            return lane['name']
    return None

# Function to process video
def process_video(source, lanes_file, save_video, save_csv):
    # Load YOLOv8 model
    model = YOLO("yolov8n.pt")

    # Load lanes
    with open(lanes_file, 'r', encoding='utf-8-sig') as f:
        lanes = json.load(f)
    print("Lanes loaded:", lanes)

    # Initialize lane counters
    lane_counters = {lane['name']: 0 for lane in lanes['lanes']}

    # Open video
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"[ERROR] Cannot open video: {source}")
        return

    # Prepare output
    os.makedirs("output", exist_ok=True)
    out = None
    if save_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter("output/annotated.mp4", fourcc, 30.0,
                              (int(cap.get(3)), int(cap.get(4))))

    # CSV setup
    csv_file = open("output/vehicle_log.csv", "w", newline="") if save_csv else None
    csv_writer = csv.writer(csv_file) if save_csv else None
    if save_csv:
        csv_writer.writerow(["Frame", "VehicleID", "Class", "Confidence", "Xmin", "Ymin", "Xmax", "Ymax", "Lane"])

    frame_count = 0
    total_vehicles = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        # Run YOLO detection
        results = model(frame, verbose=False)
        for result in results:
            boxes = result.boxes
            for box in boxes:
                cls_id = int(box.cls)
                label = model.names[cls_id]
                conf = float(box.conf)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

                # Vehicle center
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                # Determine lane
                lane_name = get_lane(lanes, cx, cy)
                if lane_name:
                    lane_counters[lane_name] += 1

                # Draw text without background
                text = f"{label} {conf:.2f}"
                cv2.putText(frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_TRIPLEX, 0.7, (0, 255, 0), 2)

                total_vehicles += 1

                # Save CSV
                if save_csv:
                    csv_writer.writerow([frame_count, total_vehicles, label, f"{conf:.2f}", x1, y1, x2, y2, lane_name])

        # Display lane counters
        y_offset = 30
        for lane_name, count in lane_counters.items():
            cv2.putText(frame, f"{lane_name}: {count}", (10, y_offset),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            y_offset += 30

        # Write video
        if save_video:
            out.write(frame)

        cv2.imshow("Traffic Flow Analysis", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    if out:
        out.release()
    if csv_file:
        csv_file.close()
    cv2.destroyAllWindows()

    # Save summary
    summary = {"Total Frames": frame_count, "Total Vehicles": total_vehicles, "Lane Counts": lane_counters}
    with open("output/summary.json", "w") as f:
        json.dump(summary, f, indent=4)

    print("\n--- HR Traffic Report ---")
    print(f"Total frames: {frame_count}")
    print(f"Total vehicles detected: {total_vehicles}")
    print("Vehicle log saved to output/vehicle_log.csv")
    print("Summary saved to output/summary.json")
    if save_video:
        print("Annotated video saved to output/annotated.mp4")

# Main
if __name__ == "__main__":
    import sys

    parser = argparse.ArgumentParser(description="Traffic Flow Analysis using YOLOv8")
    parser.add_argument("--source", required=True, help="Video file path or YouTube URL")
    parser.add_argument("--lanes", required=True, help="JSON file defining lanes")
    parser.add_argument("--save-video", action="store_true", help="Save annotated video")
    parser.add_argument("--csv", action="store_true", help="Save vehicle log as CSV")

    # Use default test values if no arguments provided
    if len(sys.argv) == 1:
        args = parser.parse_args([
            "--source", "https://www.youtube.com/watch?v=MNn9qKG2UFI",
            "--lanes", "lanes.json",
            "--save-video",
            "--csv"
        ])
    else:
        args = parser.parse_args()

    # Download video if URL
    video_source = download_video(args.source)

    process_video(video_source, args.lanes, args.save_video, args.csv)
