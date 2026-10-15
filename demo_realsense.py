#!/usr/bin/env python3
"""Run official WiLoR on a live Intel RealSense RGB stream."""

import argparse
import json
import socket
import time
from pathlib import Path

import cv2
import numpy as np
import pyrealsense2 as rs
import torch
from ultralytics import YOLO

from wilor.datasets.vitdet_dataset import ViTDetDataset
from wilor.models import load_wilor
from wilor.utils import recursive_to
from wilor.utils.renderer import cam_crop_to_full


MESH_COLOR = (168, 70, 64)  # BGR, purple


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", help="RealSense serial number; auto-select if only one is connected")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--hand-conf", type=float, default=0.3)
    parser.add_argument("--fast", action="store_true", help="Use FP16 and WiLoR depth pruning")
    parser.add_argument("--cpu", action="store_true", help="Disable CUDA")
    parser.add_argument("--udp-host", default="127.0.0.1", help="Host receiving WiLoR 21-joint packets")
    parser.add_argument("--udp-port", type=int, default=5005, help="UDP port; set 0 to disable publishing")
    return parser.parse_args()


def require_assets():
    missing = [
        path for path in (
            Path("pretrained_models/detector.pt"),
            Path("pretrained_models/wilor_final.ckpt"),
            Path("mano_data/MANO_RIGHT.pkl"),
        ) if not path.exists()
    ]
    if missing:
        paths = "\n  ".join(str(path) for path in missing)
        raise SystemExit(
            "Missing WiLoR model asset(s):\n  " + paths
            + "\nFollow the download instructions in README.md. MANO_RIGHT.pkl requires the MANO licence."
        )


def draw_mesh(frame, verts, cam_t, faces, focal, width, height):
    """Project MANO triangles and blend their silhouette onto a BGR frame."""
    vertices = verts + cam_t[None]
    z = vertices[:, 2]
    valid = np.isfinite(vertices).all(axis=1) & (z > 1e-4)
    projected = np.empty((len(vertices), 2), dtype=np.float32)
    projected[:, 0] = focal * vertices[:, 0] / z + width / 2
    projected[:, 1] = focal * vertices[:, 1] / z + height / 2
    triangles = faces[valid[faces].all(axis=1)]
    triangles = triangles[np.argsort(z[triangles].mean(axis=1))[::-1]]
    overlay = frame.copy()
    for triangle in triangles:
        points = np.rint(projected[triangle]).astype(np.int32)
        if np.abs(points).max() <= 10000:
            cv2.fillConvexPoly(overlay, points, MESH_COLOR, lineType=cv2.LINE_AA)
    return cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)


def main():
    args = parse_args()
    require_assets()
    if args.serial is None:
        devices = list(rs.context().query_devices())
        if len(devices) != 1:
            raise SystemExit(
                f"Found {len(devices)} RealSense cameras; connect one or pass --serial SERIAL."
            )
        args.serial = devices[0].get_info(rs.camera_info.serial_number)
    camera = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    try:
        profile = camera.start(config)
    except RuntimeError as error:
        raise SystemExit(f"Cannot start RealSense {args.serial}: {error}") from error

    intr = profile.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
    focal = (intr.fx + intr.fy) / 2 * 256 / max(args.width, args.height)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"RealSense {args.serial}: {args.width}x{args.height}@{args.fps}; device={device}")
    print("Loading official WiLoR. Press q or Esc to quit.")
    model, model_cfg = load_wilor(
        "pretrained_models/wilor_final.ckpt",
        "pretrained_models/model_config.yaml",
        init_renderer=False,
    )
    model_cfg.defrost()
    model_cfg.EXTRA.FOCAL_LENGTH = focal
    model_cfg.freeze()
    if args.fast and device.type == "cuda":
        model = model.half()
        model.backbone.skip_blocks = True
    model = model.eval().to(device)
    detector = YOLO("pretrained_models/detector.pt").to(device)
    faces = model.mano.faces.astype(np.int32)
    publisher = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.udp_port else None
    if publisher:
        print(f"Publishing right-hand 21-joint packets to udp://{args.udp_host}:{args.udp_port}")

    try:
        while True:
            color_frame = camera.wait_for_frames().get_color_frame()
            if not color_frame:
                continue
            frame = np.asanyarray(color_frame.get_data())
            start = time.perf_counter()
            detections = detector(frame, conf=args.hand_conf, verbose=False)[0]
            boxes, right = [], []
            for detection in detections:
                box = detection.boxes.data.cpu().squeeze().numpy()
                boxes.append(box[:4])
                right.append(detection.boxes.cls.cpu().squeeze().item())
            rendered = frame
            if boxes:
                dataset = ViTDetDataset(model_cfg, frame, np.stack(boxes), np.asarray(right),
                                        rescale_factor=2.0, fp16=args.fast and device.type == "cuda")
                batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=16, num_workers=0)))
                batch = recursive_to(batch, device)
                with torch.no_grad():
                    output = model(batch)
                pred_cam = output["pred_cam"]
                pred_cam[:, 1] *= 2 * batch["right"] - 1
                cam_t = cam_crop_to_full(pred_cam, batch["box_center"].float(), batch["box_size"].float(),
                                         batch["img_size"].float(), focal).cpu().numpy()
                for index in range(output["pred_vertices"].shape[0]):
                    verts = output["pred_vertices"][index].cpu().numpy()
                    is_right = bool(batch["right"][index].item())
                    verts[:, 0] *= 2 * is_right - 1
                    rendered = draw_mesh(rendered, verts, cam_t[index], faces, focal, args.width, args.height)
                    # XHand is a right hand.  Do not command it from a
                    # mirrored left-hand estimate.
                    if publisher and is_right:
                        joints = output["pred_keypoints_3d"][index].cpu().numpy()
                        joints[:, 0] *= 2 * is_right - 1
                        packet = json.dumps({
                            "keypoints_3d": joints.tolist(),
                            "frame": int(color_frame.get_frame_number()),
                            "timestamp_ms": float(color_frame.get_timestamp()),
                        }, separators=(",", ":")).encode()
                        publisher.sendto(packet, (args.udp_host, args.udp_port))
            elapsed = time.perf_counter() - start
            cv2.putText(rendered, f"Official WiLoR  {1 / max(elapsed, 1e-6):.1f} FPS  hands: {len(boxes)}",
                        (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Official WiLoR | RealSense", rendered)
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break
    finally:
        if publisher:
            publisher.close()
        camera.stop()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
