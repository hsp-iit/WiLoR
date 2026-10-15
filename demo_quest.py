#!/usr/bin/env python3
"""Preview a Quest RGB camera or run WiLoR on its live H.264 stream."""

import argparse
import json
import socket
import time
from pathlib import Path

import cv2
import numpy as np

from quest_camera import QuestCamera


ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-only", action="store_true", help="View camera without loading WiLoR")
    parser.add_argument("--eye", choices=("left", "right"), default="left")
    parser.add_argument("--serial", help="ADB serial, including IP:port for an authorized Wi-Fi device")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--fps", type=int, choices=(15, 30, 60), default=30)
    parser.add_argument("--bitrate", type=int, default=8_000_000, help="H.264 bits per second")
    parser.add_argument("--fast", action="store_true", help="FP16 and backbone pruning on CUDA")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--hand-conf", type=float, default=0.3)
    parser.add_argument("--udp-host", default="127.0.0.1")
    parser.add_argument("--udp-port", type=int, default=0, help="Right-hand joints output; 0 disables publishing")
    parser.add_argument("--headless", action="store_true", help="Do not open a display window")
    parser.add_argument("--duration", type=float, default=0, help="Stop after this many streaming seconds; 0 is unlimited")
    parser.add_argument("--max-frame-age-ms", type=float, default=500,
                        help="Drop results older than this since PC decode (not total sensor latency)")
    args = parser.parse_args()
    if min(args.width, args.height, args.bitrate, args.max_frame_age_ms) <= 0 or args.duration < 0:
        parser.error("Dimensions, bitrate and maximum age must be positive; duration must be nonnegative")
    if not 0 <= args.udp_port <= 65535:
        parser.error("UDP port must be 0..65535")
    return args


class HandEstimator:
    def __init__(self, fast=False, cpu=False, confidence=0.3):
        # Camera preview deliberately has no torch or MANO dependency.
        import torch
        from ultralytics import YOLO
        from wilor.models import load_wilor

        for name in ("pretrained_models/detector.pt", "pretrained_models/wilor_final.ckpt",
                     "pretrained_models/model_config.yaml", "mano_data/MANO_RIGHT.pkl"):
            if not (ROOT / name).is_file():
                raise FileNotFoundError(f"Missing {name}; see README.md for model installation")
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
        self.fp16 = fast and self.device.type == "cuda"
        self.confidence = confidence
        print(f"Loading WiLoR on {self.device} (FP16={self.fp16})", flush=True)
        self.model, self.cfg = load_wilor(str(ROOT / "pretrained_models/wilor_final.ckpt"),
                                          str(ROOT / "pretrained_models/model_config.yaml"),
                                          init_renderer=False)
        if self.fp16:
            self.model = self.model.half()
            self.model.backbone.skip_blocks = True
        self.model = self.model.eval().to(self.device)
        self.detector = YOLO(str(ROOT / "pretrained_models/detector.pt")).to(self.device)
        self.faces = self.model.mano.faces.astype(np.int32)

    def predict(self, frame):
        import torch
        from wilor.datasets.vitdet_dataset import ViTDetDataset
        from wilor.utils import recursive_to
        from wilor.utils.renderer import cam_crop_to_full

        detections = self.detector(frame, conf=self.confidence, verbose=False)[0].boxes
        if len(detections) == 0:
            return []
        dataset = ViTDetDataset(self.cfg, frame, detections.xyxy.cpu().numpy(),
                                detections.cls.cpu().numpy(), rescale_factor=2.0, fp16=self.fp16)
        # Use the same nominal focal convention as demo.py for the overlay.
        # The output joints are model coordinates, not calibrated world positions.
        focal = self.cfg.EXTRA.FOCAL_LENGTH / self.cfg.MODEL.IMAGE_SIZE * max(frame.shape[:2])
        hands = []
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, num_workers=0)
        with torch.inference_mode():
            for batch in loader:
                batch = recursive_to(batch, self.device)
                output = self.model(batch)
                pred_cam = output["pred_cam"].float()
                pred_cam[:, 1] *= 2 * batch["right"] - 1
                translations = cam_crop_to_full(pred_cam, batch["box_center"].float(),
                                                batch["box_size"].float(), batch["img_size"].float(),
                                                focal).cpu().numpy()
                for i in range(len(translations)):
                    right = bool(batch["right"][i].item())
                    vertices = output["pred_vertices"][i].float().cpu().numpy().copy()
                    joints = output["pred_keypoints_3d"][i].float().cpu().numpy().copy()
                    vertices[:, 0] *= 2 * right - 1
                    joints[:, 0] *= 2 * right - 1
                    if np.isfinite(joints).all():
                        hands.append(dict(right=right, joints=joints, vertices=vertices,
                                          translation=translations[i], focal=focal))
        return hands

    def draw(self, frame, hands):
        overlay = frame.copy()
        height, width = frame.shape[:2]
        for hand in hands:
            vertices = hand["vertices"] + hand["translation"]
            valid = np.isfinite(vertices).all(axis=1) & (vertices[:, 2] > 1e-4)
            points = np.zeros((len(vertices), 2), dtype=np.float32)
            points[valid] = vertices[valid, :2] / vertices[valid, 2:3] * hand["focal"] + [width / 2, height / 2]
            for triangle in self.faces[valid[self.faces].all(axis=1)]:
                projected = points[triangle]
                if np.isfinite(projected).all() and np.abs(projected).max() < 10000:
                    cv2.fillConvexPoly(overlay, np.rint(projected).astype(np.int32),
                                      (168, 70, 64) if hand["right"] else (64, 160, 80), cv2.LINE_AA)
        return cv2.addWeighted(overlay, 0.55, frame, 0.45, 0)


def main():
    args = parse_args()
    estimator = None if args.preview_only else HandEstimator(args.fast, args.cpu, args.hand_conf)
    publisher = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) if args.udp_port and estimator else None
    if publisher:
        print(f"Publishing right-hand joints to udp://{args.udp_host}:{args.udp_port}", flush=True)
    window = "Quest RGB" if args.preview_only else "WiLoR | Quest RGB"
    count = hand_frames = 0
    last_number = 0
    try:
        with QuestCamera(eye=args.eye, width=args.width, height=args.height, fps=args.fps,
                         bitrate=args.bitrate, serial=args.serial) as camera:
            print(f"Quest {args.eye} RGB camera {camera.camera_id}: {args.width}x{args.height}@{args.fps}, "
                  f"H.264 {args.bitrate / 1e6:g} Mbps. Press q or Esc to stop.", flush=True)
            if not args.headless:
                cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window, 960, round(960 * args.height / args.width))
            started = time.monotonic()
            while not args.duration or time.monotonic() - started < args.duration:
                received = camera.read(after=last_number)
                last_number = received.number
                frame = received.bgr
                tick = time.monotonic()
                hands = estimator.predict(frame) if estimator else []
                age_ms = (time.monotonic_ns() - received.received_monotonic_ns) / 1e6
                stale = age_ms > args.max_frame_age_ms
                count += 1
                hand_frames += bool(hands) and not stale
                if publisher and not stale:
                    for hand in hands:
                        if hand["right"]:
                            publisher.sendto(json.dumps({
                                "keypoints_3d": hand["joints"].tolist(),
                                "frame": received.number,
                                "timestamp_ms": received.received_monotonic_ns / 1e6,
                                "timestamp_domain": "pc_monotonic_decode",
                                "source": "quest_rgb_wilor",
                            }, separators=(",", ":")).encode(), (args.udp_host, args.udp_port))
                if not args.headless:
                    rendered = estimator.draw(frame, hands) if estimator and not stale else frame.copy()
                    status = f"{args.eye} | frame {received.number} | hands {len(hands)} | processing {1000*(time.monotonic()-tick):.0f} ms"
                    if stale:
                        status += " | result too old, not published"
                    cv2.putText(rendered, status, (12, 30), cv2.FONT_HERSHEY_SIMPLEX,
                                0.65, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(window, rendered)
                    if cv2.waitKey(1) & 0xFF in (27, ord("q")) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                        break
            elapsed = time.monotonic() - started
            print(f"Processed {count} fresh frames in {elapsed:.1f}s ({count/max(elapsed, 1e-6):.1f} FPS); "
                  f"{hand_frames} frames with hands. Last received frame: {last_number}.", flush=True)
    finally:
        if publisher:
            publisher.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
    except (RuntimeError, TimeoutError, FileNotFoundError) as error:
        raise SystemExit(str(error)) from error
