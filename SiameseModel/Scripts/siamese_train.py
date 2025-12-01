"""
siamese_train.py
=================

This script trains a Siamese neural network for signature verification using
contrastive loss. It relies on the dataset defined in
``siamese_dataset.py`` and uses the base ``SignatureNet`` model defined in
``model.py``. The training parameters (epochs, batch size, learning rate,
etc.) are configurable at the top of the script.

Usage::

    python siamese_train.py

The trained model will be saved into the ``models`` directory. You can then
evaluate its performance on a validation set with ``siamese_evaluate.py``
and run interactive comparisons via ``siamese_inference.py``.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from model import SignatureNet  # use the same backbone for Siamese training
from siamese_dataset import SignatureSiameseDataset


class ContrastiveLoss(nn.Module):
    """Contrastive loss function for Siamese networks.

    This implementation uses the formulation from Hadsell et al. (2006):

        L = (1 - Y) * D^2 + Y * max(0, margin - D)^2

    where D is the Euclidean distance between two embeddings, and Y is the
    binary label (0 for similar/genuine pairs and 1 for dissimilar/negative
    pairs). Margin defines the minimum distance expected between negative
    pairs.
    """

    def __init__(self, margin: float = 1.0) -> None:
        super().__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Compute pairwise distance between embeddings
        distances = F.pairwise_distance(output1, output2)
        # Contrastive loss: label==0 => similar => minimize distance^2,
        # label==1 => dissimilar => minimize (margin - distance)^2 when distance < margin
        loss = (1 - label) * torch.pow(distances, 2) + \
               label * torch.pow(torch.clamp(self.margin - distances, min=0.0), 2)
        return loss.mean()


def train_siamese(
    root_dir: str = '../sign_data/split/train',
    save_path: str = '../models/signature_siamese_last.pth',
    epochs: int = 90,
    batch_size: int = 16,
    learning_rate: float = 5e-4,
    margin: float = 2.0,
    pos_fraction: float = 0.5
) -> None:
    """Train a Siamese network on the given dataset.

    Args:
        root_dir: Path to the training data. Should contain user folders and
            optional ``*_forg`` folders.
        save_path: File path to store the *last epoch* model weights.
        epochs: Number of training epochs.
        batch_size: Mini‐batch size used during training.
        learning_rate: Optimizer learning rate.
        margin: Margin hyperparameter for the contrastive loss.
        pos_fraction: Fraction of samples that should be positive pairs.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    # Data augmentation similar to triplet training
    train_transforms = transforms.Compose([
        transforms.Resize((128, 224)),
        transforms.RandomRotation(degrees=10),
        transforms.RandomAffine(degrees=0, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    dataset = SignatureSiameseDataset(
        root_dir=root_dir,
        transform=train_transforms,
        pos_fraction=pos_fraction
    )
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    model = SignatureNet().to(device)
    criterion = ContrastiveLoss(margin=margin)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting Siamese network training...")
    model.train()

    # Best model takibi için
    best_loss = float('inf')
    # best model'i last ile aynı klasöre koyalım
    best_path = os.path.join(os.path.dirname(save_path), 'signature_siamese_best.pth')

    try:
        for epoch in range(epochs):
            total_loss = 0.0
            total_pos_dist = 0.0
            total_neg_dist = 0.0
            pos_count = 0
            neg_count = 0

            for batch in dataloader:
                img1, img2, labels = batch
                img1 = img1.to(device)
                img2 = img2.to(device)
                labels = labels.float().to(device)  # ensure float for computation

                optimizer.zero_grad()
                emb1 = model(img1)
                emb2 = model(img2)

                loss = criterion(emb1, emb2, labels)
                loss.backward()
                optimizer.step()

                # accumulate statistics
                total_loss += loss.item()
                with torch.no_grad():
                    distances = F.pairwise_distance(emb1, emb2)
                    for dist, lbl in zip(distances, labels):
                        if lbl.item() == 0:
                            total_pos_dist += dist.item()
                            pos_count += 1
                        else:
                            total_neg_dist += dist.item()
                            neg_count += 1

            # compute average values for this epoch
            avg_loss = total_loss / len(dataloader)
            avg_pos = total_pos_dist / max(pos_count, 1)
            avg_neg = total_neg_dist / max(neg_count, 1)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{epochs}] - Loss: {avg_loss:.4f} | "
                      f"PosDist: {avg_pos:.4f} | NegDist: {avg_neg:.4f}")

            # En iyi modeli loss'a göre kaydet
            if avg_loss < best_loss:
                best_loss = avg_loss
                os.makedirs(os.path.dirname(best_path), exist_ok=True)
                torch.save(model.state_dict(), best_path)
                print(f"--> Yeni EN İYİ model kaydedildi! Epoch {epoch+1}, Loss: {best_loss:.4f}")

    except KeyboardInterrupt:
        print("\n⛔ Eğitim kullanıcı tarafından erken durduruldu (CTRL+C).")

    # Son epoch modelini kaydet
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Training finished. Last epoch model saved to {save_path}")
    if best_loss < float('inf'):
        print(f"Best epoch model (min loss {best_loss:.4f}) saved to {best_path}")
    else:
        print("Uyarı: Hiç best model kaydedilmedi (muhtemelen dataloader hiç iterasyon üretmedi).")


if __name__ == '__main__':
    # Launch training with default hyperparameters. Users may edit values below.
    train_siamese()
