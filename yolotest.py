import argparse
import cv2
import numpy as np
import pyrealsense2 as rs
from ultralytics import YOLO

def draw_detections(frame, results):
    for result in results:
        boxes = result.boxes
        if boxes is not None:
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = f"{result.names[cls]} {conf:.2f}"
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, label, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

def draw_obb(frame, results):
    for result in results:
        obb = result.obb
        if obb is not None:
            for i in range(len(obb)):
                pts = obb.xyxyxyxy[i].cpu().numpy().reshape(4, 2).astype(int)
                conf = float(obb.conf[i])
                cls = int(obb.cls[i])
                label = f"{result.names[cls]} {conf:.2f}"
                cv2.polylines(frame, [pts], isClosed=True, color=(0, 200, 255), thickness=2)
                cv2.putText(frame, label, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", required=True, help="Path to YOLO model weights")
    args = parser.parse_args()

    model = YOLO(args.weights)
    is_obb = model.task == "obb"
    print(f"Loaded model task: {model.task}")

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    print("Press 'q' to quit.")
    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            frame = np.asanyarray(color_frame.get_data())
            results = model(frame, verbose=False)

            if is_obb:
                draw_obb(frame, results)
            else:
                draw_detections(frame, results)

            cv2.imshow("YOLO", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
