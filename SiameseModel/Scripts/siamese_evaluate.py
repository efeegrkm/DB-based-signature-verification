"""
siamese_evaluate.py
===================

Evaluate a Siamese signature verification model on a validation or test set.

This script loads a trained Siamese model (weights only) and computes
Euclidean distances between a reference signature and other signatures
belonging to the same user as well as forgeries. The distances and labels
are used to either evaluate the model at a given threshold or search for
an optimal threshold that maximizes accuracy.

Usage::

    python siamese_evaluate.py --model models/signature_siamese.pth \
        --data sign_data/split/val

You can optionally specify ``--threshold`` to evaluate at a fixed threshold.
If omitted, the script will sweep over the observed distance range and
select the threshold yielding the highest accuracy.
"""

import argparse
import os
from typing import List, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms

from model import SignatureNet


def load_model(model_path: str) -> Tuple[SignatureNet, torch.device]:
    """Load the trained Siamese model from disk and set it to evaluation mode."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = SignatureNet().to(device)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device


def prepare_transform() -> transforms.Compose:
    """Return the image preprocessing transform used for evaluation."""
    return transforms.Compose([
        transforms.Resize((128, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])


def compute_distances(
    model: SignatureNet,
    device: torch.device,
    data_dir: str
) -> Tuple[List[float], List[int]]:
    """Compute distances and labels between reference and test signatures.

    Args:
        model: Trained backbone network.
        device: Device to run computations on.
        data_dir: Directory containing user folders and optional ``*_forg``
            directories.

    Returns:
        A tuple ``(distances, labels)`` where ``distances`` is a list of
        Euclidean distances and ``labels`` is a list of integers (1 for
        genuine pairs, 0 for imposter pairs).
    """
    transform = prepare_transform()

    distances: List[float] = []
    labels: List[int] = []

    users = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d)) and not d.endswith('_forg')]
    for user in users:
        real_dir = os.path.join(data_dir, user)
        forg_dir = os.path.join(data_dir, user + '_forg')

        # gather real and forgery images
        real_imgs = [os.path.join(real_dir, f) for f in os.listdir(real_dir)
                     if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        forg_imgs: List[str] = []
        if os.path.exists(forg_dir):
            forg_imgs = [os.path.join(forg_dir, f) for f in os.listdir(forg_dir)
                         if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]

        # need at least two real images to compare
        if len(real_imgs) < 2:
            continue

        # use the first real signature as the reference
        ref_path = real_imgs[0]
        ref_img = Image.open(ref_path).convert('L')
        ref_tensor = transform(ref_img).unsqueeze(0).to(device)
        with torch.no_grad():
            ref_emb = model(ref_tensor)

        # genuine comparisons (real vs other real)
        for other_path in real_imgs[1:]:
            img = Image.open(other_path).convert('L')
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(img_tensor)
            dist = F.pairwise_distance(ref_emb, emb).item()
            distances.append(dist)
            labels.append(1)  # positive

        # imposter comparisons (real vs forgery)
        for forg_path in forg_imgs:
            img = Image.open(forg_path).convert('L')
            img_tensor = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                emb = model(img_tensor)
            dist = F.pairwise_distance(ref_emb, emb).item()
            distances.append(dist)
            labels.append(0)  # negative

    return distances, labels


def find_best_threshold(distances: List[float], labels: List[int]) -> Tuple[float, float]:
    """Search for the threshold that maximizes accuracy.

    Args:
        distances: List of distances between reference and test signatures.
        labels: List of true labels (1 for genuine, 0 for imposter).

    Returns:
        A tuple ``(best_threshold, best_accuracy)``.
    """
    if not distances:
        return 0.0, 0.0
    # candidate thresholds: unique sorted distances, plus endpoints slightly extended
    unique_dists = sorted(set(distances))
    # Add small buffer on either side
    candidates = []
    if unique_dists:
        candidates.append(unique_dists[0] - 1e-6)
        for d in unique_dists:
            candidates.append(d)
        candidates.append(unique_dists[-1] + 1e-6)
    else:
        candidates = [0.0]

    best_thr = 0.0
    best_acc = 0.0
    for thr in candidates:
        # predict: distance < thr => genuine, else imposter
        tp = tn = fp = fn = 0
        for d, lbl in zip(distances, labels):
            pred = 1 if d < thr else 0
            if pred == 1 and lbl == 1:
                tp += 1
            elif pred == 0 and lbl == 0:
                tn += 1
            elif pred == 1 and lbl == 0:
                fp += 1
            elif pred == 0 and lbl == 1:
                fn += 1
        total = len(labels)
        acc = (tp + tn) / total if total > 0 else 0.0
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    return best_thr, best_acc


def evaluate(
    model_path: str,
    data_dir: str,
    threshold: float = None
) -> None:
    """Evaluate the Siamese model on a dataset.

    If ``threshold`` is provided, use it to compute metrics. Otherwise find
    the threshold that maximizes accuracy on the data.
    """
    model, device = load_model(model_path)
    print(f"Loaded model from {model_path}")
    distances, labels = compute_distances(model, device, data_dir)
    if not distances:
        print("No comparisons were generated. Make sure your dataset contains at least two genuine signatures per user.")
        return

    if threshold is None:
        thr, best_acc = find_best_threshold(distances, labels)
        print(f"Best threshold determined: {thr:.4f} with accuracy {best_acc:.4f}")
        threshold = thr
    else:
        # When threshold is provided we can compute accuracy directly
        thr = threshold

    # Compute metrics at the chosen threshold
    tp = tn = fp = fn = 0
    for d, lbl in zip(distances, labels):
        pred = 1 if d < threshold else 0
        if pred == 1 and lbl == 1:
            tp += 1
        elif pred == 0 and lbl == 0:
            tn += 1
        elif pred == 1 and lbl == 0:
            fp += 1
        elif pred == 0 and lbl == 1:
            fn += 1
    total = len(labels)
    accuracy = (tp + tn) / total
    print("Evaluation results:")
    print(f"  Total comparisons: {total}")
    print(f"  True Positives:    {tp}")
    print(f"  True Negatives:    {tn}")
    print(f"  False Positives:   {fp}")
    print(f"  False Negatives:   {fn}")
    print(f"  Accuracy:          {accuracy:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate a Siamese signature model.")
    parser.add_argument('--model', required=True, help='Path to the trained Siamese model weights.')
    parser.add_argument('--data', required=True, help='Path to the evaluation data directory (e.g. sign_data/split/val).')
    parser.add_argument('--threshold', type=float, default=None, help='Fixed threshold to use for classification. If omitted, the script will determine the best threshold.')
    args = parser.parse_args()
    evaluate(args.model, args.data, args.threshold)


if __name__ == '__main__':
    main()