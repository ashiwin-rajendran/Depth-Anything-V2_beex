#!/usr/bin/env python3

import torch.nn.functional as F

# ==== MONKEY-PATCH interpolate to drop 'antialias' on older PyTorch ====
_orig_interp = F.interpolate


def _safe_interpolate(input, *args, **kwargs):
    kwargs.pop("antialias", None)
    return _orig_interp(input, *args, **kwargs)


F.interpolate = _safe_interpolate
# =======================================================================

import argparse
import cv2
import glob

# import matplotlib
import cv2
import numpy as np
import os
import torch
import rospy
from sensor_msgs.msg import CompressedImage

from depth_anything_v2.dpt import DepthAnythingV2


def main():
    parser = argparse.ArgumentParser(description="Depth Anything V2 ROS Node")
    parser.add_argument(
        "--video-path",
        type=str,
        default="/ikan/ip_cam/ml_clahe/compressed",
        help="Input ROS topic for sensor_msgs/CompressedImage (e.g. /camera/image/compressed)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="/ikan/depth/image/compressed",
        help="Output ROS topic for sensor_msgs/CompressedImage",
    )
    parser.add_argument("--input-size", type=int, default=217)
    parser.add_argument("--encoder", type=str, default="vits", choices=["vits", "vitb", "vitl", "vitg"])
    parser.add_argument(
        "--pred-only", dest="pred_only", action="store_true", help="only publish the prediction (no side‑by‑side)"
    )
    parser.add_argument("--grayscale", dest="grayscale", action="store_true", help="do not apply colorful palette")
    args = parser.parse_args()

    rospy.init_node("depth_anything_v2_node", anonymous=True)

    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # ─── Existing model config dict ──────────────────────────────
    model_configs = {
        "vits": {"encoder": "vits", "features": 64, "out_channels": [48, 96, 192, 384]},
        "vitb": {"encoder": "vitb", "features": 128, "out_channels": [96, 192, 384, 768]},
        "vitl": {"encoder": "vitl", "features": 256, "out_channels": [256, 512, 1024, 1024]},
        "vitg": {"encoder": "vitg", "features": 384, "out_channels": [1536, 1536, 1536, 1536]},
    }

    depth_anything = DepthAnythingV2(**model_configs[args.encoder])
    depth_anything.load_state_dict(torch.load(f"checkpoints/depth_anything_v2_{args.encoder}.pth", map_location="cpu"))
    depth_anything = depth_anything.to(DEVICE).eval()

    margin_width = 50
    cv_colormap = cv2.COLORMAP_JET

    #### Old Method - Using matplot color maps
    # cmap = matplotlib.colormaps.get_cmap("Spectral_r")

    # Publisher for output depth images
    pub = rospy.Publisher(args.outdir, CompressedImage, queue_size=1)

    def callback(msg: CompressedImage):
        try:
            # Decode incoming CompressedImage to OpenCV BGR
            np_arr = np.frombuffer(msg.data, np.uint8)
            raw_frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)

            # ─── Existing inference & coloring code ─────────────────
            depth = depth_anything.infer_image(raw_frame, args.input_size)
            depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
            depth = depth.astype(np.uint8)

            if args.grayscale:
                # simple gray→3-channel
                depth_bgr = np.repeat(depth[..., None], 3, axis=-1)
            else:
                # use OpenCV to colorize
                # (input must be 8-bit single channel; output is BGR)
                depth_bgr = cv2.applyColorMap(depth, cv_colormap)

            ### Older method - Using Matplotlib
            # if args.grayscale:
            #     depth_bgr = np.repeat(depth[..., None], 3, axis=-1)
            # else:
            #     depth_bgr = (cmap(depth)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            #########

            if args.pred_only:
                output_frame = depth_bgr
            else:
                h, w = raw_frame.shape[:2]
                split_region = np.ones((h, margin_width, 3), np.uint8) * 255
                output_frame = cv2.hconcat([raw_frame, split_region, depth_bgr])

            # Encode back to JPEG for CompressedImage
            _, buffer = cv2.imencode(".jpg", output_frame)
            out_msg = CompressedImage()
            out_msg.header = msg.header
            out_msg.format = "jpeg"
            out_msg.data = np.array(buffer).tobytes()

            pub.publish(out_msg)

        except Exception as e:
            rospy.logerr(f"[depth_anything_v2] processing error: {e}")

    # Subscribe to the input topic
    rospy.Subscriber(args.video_path, CompressedImage, callback, queue_size=1, buff_size=2**24)

    rospy.spin()


if __name__ == "__main__":
    main()
