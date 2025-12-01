import torch
import torch.nn.functional as F
from torchvision import transforms
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
# ------------------------------
# 1️⃣ Config — Dizin ve Model
# ------------------------------
MODEL_PATH = r"C:\Users\efegr\OneDrive\Belgeler\PythonProjects\SignatureAuthentication\outputs\checkpoints\FirstEpoch96Val\best_model_epoch1.pth"
TEST_ROOT = r"C:\Users\efegr\OneDrive\Belgeler\PythonProjects\SignatureAuthentication\data\final_data\test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# 2️⃣ Görüntü transformları
# ------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ------------------------------
# 3️⃣ Yardımcı Fonksiyonlar
# ------------------------------
def load_image(path):
    img = Image.open(path).convert("L")
    return transform(img).unsqueeze(0).to(DEVICE)

def get_all_pairs(genuine_dir, forgery_dir):
    genuine_images = [os.path.join(genuine_dir, f) for f in os.listdir(genuine_dir) if f.endswith(".png")]
    forgery_images = [os.path.join(forgery_dir, f) for f in os.listdir(forgery_dir) if f.endswith(".png")]

    pairs, labels = [], []

    for g in genuine_images:
        for g2 in genuine_images:
            if g != g2:
                pairs.append((g, g2))
                labels.append(1)  # Genuine pair
        for f in forgery_images:
            pairs.append((g, f))
            labels.append(0)  # Forgery pair
    return pairs, labels

# ------------------------------
# 4️⃣ Modeli Yükle
# ------------------------------
model = torch.load(MODEL_PATH, map_location=DEVICE)
model.eval()

# Eğer model state_dict olarak kaydedildiyse:
# model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
# model.to(DEVICE)

# ------------------------------
# 5️⃣ Test Pairlerini Oluştur
# ------------------------------
genuine_dir = os.path.join(TEST_ROOT, "genuine")
forgery_dir = os.path.join(TEST_ROOT, "forgery")
pairs, labels = get_all_pairs(genuine_dir, forgery_dir)

print(f"Toplam test pair sayısı: {len(pairs)}")

# ------------------------------
# 6️⃣ Test Döngüsü
# ------------------------------
scores = []
with torch.no_grad():
    for img1_path, img2_path in pairs:
        img1 = load_image(img1_path)
        img2 = load_image(img2_path)

        emb1 = model(img1)
        emb2 = model(img2)

        # Cosine similarity kullanıyoruz
        score = F.cosine_similarity(emb1, emb2).item()
        scores.append(score)

# ------------------------------
# 7️⃣ Metrikleri Hesapla
# ------------------------------
scores = np.array(scores)
labels = np.array(labels)

auc = roc_auc_score(labels, scores)

# ROC ve EER hesapla
fpr, tpr, thresholds = roc_curve(labels, scores)
fnr = 1 - tpr
eer_threshold = thresholds[np.nanargmin(np.absolute((fnr - fpr)))]
eer = fpr[np.nanargmin(np.absolute((fnr - fpr)))]

# Binary accuracy için threshold uygula
preds = (scores > eer_threshold).astype(int)
acc = accuracy_score(labels, preds)
prec = precision_score(labels, preds)
rec = recall_score(labels, preds)
f1 = f1_score(labels, preds)

print(f"\n✅ TEST SONUÇLARI ✅")
print(f"ROC-AUC       : {auc:.4f}")
print(f"EER            : {eer:.4f}")
print(f"Accuracy       : {acc:.4f}")
print(f"Precision      : {prec:.4f}")
print(f"Recall         : {rec:.4f}")
print(f"F1-score       : {f1:.4f}")
print(f"Optimal thr    : {eer_threshold:.4f}")

# ------------------------------
# 8️⃣ ROC Eğrisi (isteğe bağlı)
# ------------------------------
plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"ROC curve (AUC={auc:.3f})")
plt.plot([0,1],[0,1],'--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve — Signature Verification")
plt.legend()
plt.show()
