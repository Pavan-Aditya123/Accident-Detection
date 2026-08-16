"""
Zero-DCE Low-Light Image & Video Enhancement Inference Module.

Pipeline Integration:
- Loads best trained weights (best_zero_dce.pth) from models/enhancement/zero_dce/.
- Accepts low-light image frames, NumPy arrays (OpenCV BGR), or directory batches.
- Returns enhanced frames for downstream accident detection (e.g., YOLO26).
- Saves inference outputs to results/enhancement/zero_dce/.
"""

import sys
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT_DIR))

from glob import glob
import cv2
import numpy as np

import torch

from models.enhancement.zero_dce.model import DCENet


class ImageEnhancer:
    """
    Zero-DCE Low-Light Image Enhancer.
    Loads trained DCE-Net weights and performs brightness enhancement.
    """

    def __init__(self, model_path: str = None, device: str = None):

        self.base_dir = Path(__file__).resolve().parent.parent

        if model_path is None:
            model_path = (
                self.base_dir
                / "models"
                / "enhancement"
                / "zero_dce"
                / "best_zero_dce.pth"
            )

        self.model_path = Path(model_path)

        if device is None:
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() else "cpu"
            )
        else:
            self.device = torch.device(device)

        self.model = DCENet().to(self.device)
        self.load_weights()


    def load_weights(self):

        if self.model_path.exists():

            checkpoint = torch.load(
                self.model_path,
                map_location=self.device
            )

            self.model.load_state_dict(checkpoint)
            self.model.eval()

            print(
                f"[+] Loaded Zero-DCE model weights from: '{self.model_path}'"
            )

        else:

            print(
                f"[!] Warning: Model weight file not found at '{self.model_path}'"
            )
            self.model.eval()


    def enhance_tensor(self, img_tensor):

        with torch.no_grad():

            if img_tensor.dim() == 3:
                img_tensor = img_tensor.unsqueeze(0)

            img_tensor = img_tensor.to(self.device)

            enhanced_tensor, _ = self.model(img_tensor)

            enhanced_tensor = torch.clamp(
                enhanced_tensor,
                0.0,
                1.0
            )

            return enhanced_tensor.squeeze(0)


    def enhance_cv2(self, frame_bgr):

        frame_rgb = cv2.cvtColor(
            frame_bgr,
            cv2.COLOR_BGR2RGB
        )

        img_tensor = (
            torch.from_numpy(frame_rgb)
            .float()
            .permute(2,0,1)
            / 255.0
        )

        enhanced_tensor = self.enhance_tensor(img_tensor)

        enhanced_rgb = (
            enhanced_tensor
            .permute(1,2,0)
            .cpu()
            .numpy()
            * 255.0
        ).astype(np.uint8)


        enhanced_bgr = cv2.cvtColor(
            enhanced_rgb,
            cv2.COLOR_RGB2BGR
        )

        return enhanced_bgr



    def enhance_directory(self, input_dir, output_dir=None):

        input_path = Path(input_dir)

        if output_dir is None:

            output_path = (
                self.base_dir
                / "results"
                / "enhancement"
                / "zero_dce"
            )

        else:

            output_path = Path(output_dir)


        output_path.mkdir(
            parents=True,
            exist_ok=True
        )


        # Added webp support
        image_files = (
            glob(str(input_path / "*.png")) +
            glob(str(input_path / "*.jpg")) +
            glob(str(input_path / "*.jpeg")) +
            glob(str(input_path / "*.webp"))
        )


        print(
            f"[+] Found {len(image_files)} images in '{input_path}'. Enhancing..."
        )


        for img_file in image_files:

            img = cv2.imread(img_file)

            if img is None:
                continue


            enhanced = self.enhance_cv2(img)

            output_file = (
                output_path /
                Path(img_file).with_suffix(".png").name
            )

            cv2.imwrite(
                str(output_file),
                enhanced
            )


        print(
            f"[+] Enhanced images successfully saved to: '{output_path}'"
        )



if __name__ == "__main__":


    enhancer = ImageEnhancer()


    # Test custom images from inference/images
    base_repo = Path(__file__).resolve().parent.parent

    test_images = (
        base_repo
        / "inference"
        / "images"
    )


    if test_images.exists():

        enhancer.enhance_directory(
            str(test_images)
        )

    else:

        print(
            "[!] inference/images folder not found."
        )