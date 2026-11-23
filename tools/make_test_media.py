"""Generate synthetic VisA-like PCB image sequences for the RTSP simulator.

No dataset download required. Produces data/media/seqN/frame_0000.png ...
with dark-green boards, traces, pads and moving components so the tracker has
real pixel motion to count.
"""

import argparse
import os

import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./data/media")
    ap.add_argument("--sequences", type=int, default=4)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--width", type=int, default=640)
    ap.add_argument("--height", type=int, default=480)
    args = ap.parse_args()

    rng = np.random.default_rng(7)
    for s in range(args.sequences):
        d = os.path.join(args.out, f"seq{s}")
        os.makedirs(d, exist_ok=True)
        base = np.full((args.height, args.width, 3), (26, 46, 30), np.uint8)
        for _ in range(30):
            x, y = int(rng.integers(0, args.width)), int(rng.integers(0, args.height))
            cv2.line(base, (x, y),
                     (int(rng.integers(0, args.width)), int(rng.integers(0, args.height))),
                     (38, 92, 52), int(rng.integers(1, 3)), cv2.LINE_AA)
        for _ in range(60):
            x, y = int(rng.integers(10, args.width - 10)), int(rng.integers(10, args.height - 10))
            cv2.circle(base, (x, y), int(rng.integers(2, 5)), (150, 130, 70), -1)
        cv2.rectangle(base, (4, 4), (args.width - 5, args.height - 5), (60, 110, 70), 2)

        for f in range(args.frames):
            img = base.copy()
            x = int((f * 12 + s * 90) % (args.width + 80)) - 40
            y = int(args.height * 0.45)
            cv2.rectangle(img, (x, y), (x + 52, y + 34), (60, 60, 220), -1)
            cv2.rectangle(img, (x + 44, y + 8), (x + 62, y + 26), (200, 200, 90), -1)
            cv2.imwrite(os.path.join(d, f"frame_{f:04d}.png"), img)
        print(f"wrote {d} ({args.frames} frames)")


if __name__ == "__main__":
    main()
