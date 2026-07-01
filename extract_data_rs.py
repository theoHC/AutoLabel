"""
RealSense frame extraction

  .bag  (ROS1, RealSense Viewer)  →  pyrealsense2
  .db3  (ROS2 SQLite bag)         →  rosbags
"""

import os
import numpy as np
import cv2
from pathlib import Path

COLOR_TOPIC = "/camera/camera/color/image_raw"
DEPTH_TOPIC = "/camera/camera/depth/image_rect_raw"


def _save_color(img_rgb, path):
    cv2.imwrite(str(path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def _save_depth(depth_mm, path):
    clipped = np.clip(depth_mm, 50, 800)
    normed = cv2.normalize(clipped, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(str(path), cv2.applyColorMap(normed, cv2.COLORMAP_JET))


def _make_dirs(output_dir):
    rgb_dir = Path(output_dir) / "rgb"
    depth_dir = Path(output_dir) / "depth"
    rgb_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)
    return rgb_dir, depth_dir


# ---------------------------------------------------------------------------
# ROS1 .bag  (RealSense Viewer native format)
# ---------------------------------------------------------------------------

def extract_frames_bag(bag_path, output_dir):
    import pyrealsense2 as rs

    rgb_dir, depth_dir = _make_dirs(output_dir)
    rgb_count = depth_count = 0

    pipeline = rs.pipeline()
    config = rs.config()
    rs.config.enable_device_from_file(config, str(bag_path), repeat_playback=False)
    config.enable_stream(rs.stream.color)
    config.enable_stream(rs.stream.depth)

    pipeline.start(config)
    pipeline.get_active_profile().get_device().as_playback().set_real_time(False)

    try:
        while True:
            frames = pipeline.wait_for_frames(timeout_ms=5000)

            color_frame = frames.get_color_frame()
            if color_frame:
                _save_color(np.asanyarray(color_frame.get_data()),
                            rgb_dir / f"{rgb_count:06d}.jpg")
                rgb_count += 1

            depth_frame = frames.get_depth_frame()
            if depth_frame:
                _save_depth(np.asanyarray(depth_frame.get_data()),
                            depth_dir / f"{depth_count:06d}.jpg")
                depth_count += 1

    except RuntimeError:
        pass  # raised at end of file
    finally:
        pipeline.stop()

    print(f"  Saved {rgb_count} RGB and {depth_count} depth frames -> {output_dir}")


# ---------------------------------------------------------------------------
# ROS2 .db3  (rosbag2 SQLite format)
# ---------------------------------------------------------------------------

def extract_frames_db3(bag_dir, output_dir):
    from rosbags.rosbag2 import Reader
    from rosbags.typesys import Stores, get_typestore

    typestore = get_typestore(Stores.ROS2_HUMBLE)
    rgb_dir, depth_dir = _make_dirs(output_dir)
    rgb_count = depth_count = 0

    with Reader(bag_dir) as reader:
        conns = [c for c in reader.connections
                 if c.topic in (COLOR_TOPIC, DEPTH_TOPIC)]
        for conn, _ts, rawdata in reader.messages(connections=conns):
            msg = typestore.deserialize_cdr(rawdata, conn.msgtype)

            if conn.topic == COLOR_TOPIC:
                img = (np.frombuffer(msg.data, dtype=np.uint8)
                         .reshape(msg.height, msg.width, 3))
                _save_color(img, rgb_dir / f"{rgb_count:06d}.jpg")
                rgb_count += 1

            elif conn.topic == DEPTH_TOPIC:
                depth = (np.frombuffer(msg.data, dtype=np.uint16)
                           .reshape(msg.height, msg.width))
                _save_depth(depth, depth_dir / f"{depth_count:06d}.jpg")
                depth_count += 1

    print(f"  Saved {rgb_count} RGB and {depth_count} depth frames -> {output_dir}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    SCRIPT_DIR = Path(__file__).parent
    DATA_ROOT = Path(os.environ.get("DATA_ROOT", SCRIPT_DIR / "data"))
    OUTPUT_ROOT = Path(os.environ.get("OUTPUT_ROOT", SCRIPT_DIR / "output"))

    # ROS1 .bag files
    for bag_file in DATA_ROOT.rglob("*.bag"):
        rel_path = bag_file.parent.relative_to(DATA_ROOT)
        print(f"Processing (ROS1 bag): {bag_file.name}")
        extract_frames_bag(bag_file, OUTPUT_ROOT / rel_path)

    # ROS2 .db3 bags — collect unique bag directories (handles split bags)
    db3_dirs = {f.parent for f in DATA_ROOT.rglob("*.db3")}
    for bag_dir in sorted(db3_dirs):
        rel_path = bag_dir.relative_to(DATA_ROOT)
        print(f"Processing (ROS2 db3): {bag_dir.name}")
        extract_frames_db3(bag_dir, OUTPUT_ROOT / rel_path)

    print("Done.")
