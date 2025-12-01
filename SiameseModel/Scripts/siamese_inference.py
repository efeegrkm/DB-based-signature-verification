"""
siamese_inference.py
====================

Perform one‐off inference using a trained Siamese signature verification model.

Given two image paths, the script loads the model, embeds both images and
reports the Euclidean distance between their embeddings. If a threshold is
provided, a simple classification (same signer vs. different signer) is
performed.

Example::

    python siamese_inference.py --model models/signature_siamese.pth \
        --img1 sign_data/split/test/049/01_049.png \
        --img2 sign_data/split/test/049/02_049.png \
        --threshold 1.0

"""

import argparse
import os
from typing import Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from model import SignatureNet


def load_model(model_path: str) -> Tuple[SignatureNet, torch.device]:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignatureNet().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def preprocess_image(image_path: str) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((128, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image file not found: {image_path}")
    image = Image.open(image_path).convert('L')
    return transform(image).unsqueeze(0)


def infer(model_path: str, img1_path: str, img2_path: str, threshold: float = None) -> None:
    model, device = load_model(model_path)
    img1 = preprocess_image(img1_path).to(device)
    img2 = preprocess_image(img2_path).to(device)
    with torch.no_grad():
        emb1 = model(img1)
        emb2 = model(img2)
        dist = F.pairwise_distance(emb1, emb2).item()
    print(f"Distance between embeddings: {dist:.4f}")
    if threshold is not None:
        if dist < threshold:
            result = "SAME (genuine)"
        else:
            result = "DIFFERENT (imposter)"
        print(f"Decision at threshold {threshold:.4f}: {result}")


def main():
    parser = argparse.ArgumentParser(description="Run inference with a trained Siamese signature model.")
    parser.add_argument('--model', required=True, help='Path to the trained Siamese model weights.')
    parser.add_argument('--img1', required=True, help='Path to the first image.')
    parser.add_argument('--img2', required=True, help='Path to the second image.')
    parser.add_argument('--threshold', type=float, default=None, help='Classification threshold (optional).')
    args = parser.parse_args()
    infer(args.model, args.img1, args.img2, args.threshold)


if __name__ == '__main__':
    main()