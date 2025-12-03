import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import json

from dataloader import get_train_loader
from model import SignatureNet

def train():
    # --- AYARLAR ---
    root_dir = 'triplet_model/sign_data/split/train'
    epochs = 200          
    batch_size = 16       
    learning_rate = 0.0005 
    margin = 1.2        

    # GRAFİK VERİLERİ — RAPOR İÇİN
    train_losses = []
    pos_dists = []
    neg_dists = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Eğitim şu cihazda yapılacak: {device}")

    train_loader = get_train_loader(root_dir, batch_size)
    model = SignatureNet().to(device)

    criterion = nn.TripletMarginLoss(margin=margin, p=2)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Eğitim başlıyor...")

    for epoch in range(epochs):
        running_loss = 0.0
        running_pos_dist = 0.0
        running_neg_dist = 0.0

        for anchor, positive, negative in train_loader:
            anchor = anchor.to(device)
            positive = positive.to(device)
            negative = negative.to(device)

            optimizer.zero_grad()

            emb_anchor = model(anchor)
            emb_positive = model(positive)
            emb_negative = model(negative)

            loss = criterion(emb_anchor, emb_positive, emb_negative)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Diagnostik mesafeler
            with torch.no_grad():
                pos_dist = F.pairwise_distance(emb_anchor, emb_positive).mean()
                neg_dist = F.pairwise_distance(emb_anchor, emb_negative).mean()
                running_pos_dist += pos_dist.item()
                running_neg_dist += neg_dist.item()

        # ORTALAMA METRİKLER
        avg_loss = running_loss / len(train_loader)
        avg_pos = running_pos_dist / len(train_loader)
        avg_neg = running_neg_dist / len(train_loader)

        # METRİKLERİ GRAFİK İÇİN KAYDET
        train_losses.append(avg_loss)
        pos_dists.append(avg_pos)
        neg_dists.append(avg_neg)

        # Log
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] -> Loss: {avg_loss:.4f} | PosDist: {avg_pos:.4f} | NegDist: {avg_neg:.4f}")

    # --- MODEL KAYDETME ---
    os.makedirs('models', exist_ok=True)
    save_path = 'models/signature_cnn_augmented.pth'
    torch.save(model.state_dict(), save_path)
    print(f"Eğitim bitti. Model kaydedildi: {save_path}")

    # --- METRİKLERİ JSON'E KAYDET (plot için) ---
    log_dir = 'training_logs'
    os.makedirs(log_dir, exist_ok=True)

    metrics = {
        "epochs": epochs,
        "train_loss": train_losses,
        "pos_dist": pos_dists,
        "neg_dist": neg_dists
    }

    with open(os.path.join(log_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print("Metrikler kaydedildi: training_logs/metrics.json")

if __name__ == "__main__":
    train()
