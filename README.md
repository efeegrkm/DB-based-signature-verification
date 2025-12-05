
# Direct link to modals: https://drive.google.com/drive/folders/1LX6RKq7uMcwOhMbTLXyJD8Ep2e2FRMjF?usp=sharing
# 🖋️ SignatureNetDB  
### Deep-Learning Based Signature Verification with Siamese & Triplet Networks + Identity Database

---

## 📌 Overview
**SignatureNetDB** is a full end-to-end signature verification system combining deep learning, preprocessing, and a structured user database.

This project allows:
- High-accuracy signature comparison using **Siamese** or **Triplet** embeddings  
- Storing multiple user signatures in a **database**
- Averaging signature embeddings to create a **stable identity vector**
- Verifying if:
  - A signature belongs to a specific user (via NO)
  - A queried signature exists in the database
  - Two input signatures match
- Switching between Siamese & Triplet models dynamically

---

## 🧠 Deep Learning Models

### ✔️ Siamese Network
- Contrastive Loss  
- Learns pairwise similarity  
- Best threshold found during evaluation (example): `~1.21`

### ✔️ Triplet Network
- Triplet Loss (Anchor-Positive-Negative)
- Learns better separation in embedding space  
- More robust for unseen signatures

Both models operate on:
Input shape : 1 × 400 × 400
Output embedding : 128-dimensional L2-normalized vector


---

## 🖼️ Preprocessing Pipeline (400×400)
Every signature image passes through:

1. Convert to grayscale  
2. Optional autocontrast  
3. Resize while preserving aspect ratio  
4. Center-pad into a **400×400 white canvas**  
5. Convert to tensor + normalize(`mean=0.5, std=0.5`)

The preprocessing is identical across:
- Training  
- Validation / Test evaluation  
- GUI real-time prediction  

---

## 🏋️ Training
Training scripts include:
- Hard-negative sampling  
- On-the-fly data augmentation:
  - Random rotation  
  - Small translations  
  - Light brightness/contrast jitter  

Training supports:
- Best-model saving  
- Last-checkpoint saving  
- Full loss logging  
- CUDA acceleration  

Example training output loss:  
Initial loss: ~1.00
Final best loss: ~0.13


---

## 📊 Evaluation
`siamese_evaluate.py` computes:

- All distances (genuine vs forgery)
- Optimal threshold search
- Accuracy, FP, FN, TP, TN

 Training Metrics For Main Siamese Model:
<p align="center"> <img src="./SiameseModel/logs/siamese_train_loss.png" width="420" /> <img src="./SiameseModel/logs/siamese_train_pos_neg_dist.png" width="420" /> </p>
 Evaluation Metrics
<p align="center"> <img src="./SiameseModel/logs/siamese_dist_test.png" width="420" /> <img src="./SiameseModel/logs/siamese_roc_curve.png" width="420" /> </p>
 Precision–Recall Curve
<p align="center"> <img src="./SiameseModel/logs/siamese_pr_curve.png" width="500" /> </p>

---

## 🗄️ Database System

Each registered user has:

| Field | Description |
|-------|-------------|
| **NO** | Primary Key |
| **Name** | First name |
| **Surname** | Last name |
| **Signatures** | Multiple PNG signature samples (stored in a separate table) |
| **Embedding** | Mean embedding vector of all user signatures |

### Why average the embeddings?
- Allows more stable identity representation  
- Reduces variance between signature samples  
- Works with **1 or many signatures**  

---

## 🔍 Supported Database Queries

### ✔️ 1) “Does this signature belong to user NO=X?”
- Compute embedding  
- Compare with stored user embedding  
- Apply threshold  
- Return **Genuine / Forgery**  

### ✔️ 2) “Give me NO from Name+Surname”
Simple lookup in the database.

### ✔️ 3) “Whose signature is this?”
- Compute embedding  
- Compare against **all stored embeddings**  
- Return the best match (if below threshold)

### ✔️ 4) “Verify two PNG signatures”
- Pure model-based matching  
- No database math needed  

---

## 🖥️ GUI Application

The desktop GUI includes:

- Loading two signature images  
- Real-time preprocessing visualization  
- Switching between Siamese & Triplet models  
- Distance + final decision output (color coded)  

GUI internally:
- Preprocesses images  
- Converts to tensor  
- Runs the chosen model  
- Displays both processed signatures (denormalized)
- Outputs similarity score  

---
## 🤝 Contributors
**Efe Görkem Akkanat** — Siamese Modal, GUI, Database Management.

**Şeyda Yağmur Asal** — Triplet Network, GUI, Database Management.

## Final Project Overview:
# ✒️ Signature Verification System — Model Development & Database Design Report

This document summarizes the full development process of the Siamese Signature Verification System, including model evolution, issues and solutions, evaluation results, ROC/PR analysis, training curves, and a complete database design for real-world deployment.

This markdown file is ready to be placed directly into your GitHub repository as a README or as a dedicated report.

## 1. Overview

This project implements a robust signature verification system based on:

Siamese Neural Network (Contrastive Loss)

Preprocessing pipeline for normalizing signature images

Training/Validation/Test evaluation modules

Data augmentation pipeline for real-world generalization

A database-backed identity system storing:

User metadata

Multiple raw signature images

Embedding vectors

Mean embedding for verification

The system achieves 92% test accuracy and performs strongly on real-world handwritten signatures.

## 2. Model Development Journey

Below is a detailed chronological summary of all issues encountered and how each one was resolved.

### 🚨 2.1 Issue: Incorrect Preprocessing → "Small-Patch Overfitting"

Early in development, a scaling bug caused every image to be cropped into a tiny fragment of the signature.
This had misleading effects:

Model rapidly overfitted (loss dropped too fast)

Validation/Test accuracy appeared unrealistically high

Real-world signatures performed extremely poorly

✔ Solution

Preprocessing was rewritten:

All signatures are placed on a 400×400 canvas

Scaling is corrected — full signature is always visible

Thresholding & noise cleanup standardized

🟢 Result:
Realistic generalization re-appeared, and validation accuracy dropped to a more truthful 66%, revealing the model's true state.

### 📉 2.2 Issue: Underfitting at 50 Epochs

With clean preprocessing, training for only 50 epochs was insufficient.

Symptoms:

Loss decreased slowly

Positive–negative embedding separation incomplete

Accuracy plateaued at ~66%

✔ Solution

Training parameters were updated:

Hyperparameter	Old	New
Epochs	50	90
Batch Size	16	32

🟢 Result:
Accuracy improved significantly:
66% → 92%

### 🎨 2.3 Issue: Sensitivity to Background Noise in Real Signatures

Even after 92% accuracy on the test set, real signatures still failed sometimes due to:

Background texture differences

Camera/scan brightness differences

✔ Solution: Add Color Jitter Augmentation

The following were added:

brightness=0.2

contrast=0.2

🟢 Result:
The model became background-invariant, matching real-world performance with high robustness.

## 3. Training and Evaluation Plots
### 📉 Training Loss Curve

Shows stable convergence and no overfitting.

### 🔵🟠 Positive/Negative Distance Curves

A clean growing gap indicates strong embedding separation.

### 📊 Test Distance Distribution

Genuine and forgery clusters are nearly perfectly separated.

### 📊 Validation Distance Distribution

Mirrors the test distribution, confirming generalization.

### 📈 ROC Curve (Siamese Model)

AUC ≈ 0.9936

### 📈 Precision–Recall Curve

AUC ≈ 0.9938

## 4. How the Verification Threshold Is Determined
✔ Step 1

Compute distances for all genuine pairs and forgery pairs.

✔ Step 2

Sweep thresholds over the full distance range.

✔ Step 3

Select the threshold that maximizes:

True Positive Rate

True Negative Rate

F1-score

The resulting optimal threshold:

📌 Threshold = 1.016

Evaluation results:

Metric	Value
True Positives	223
True Negatives	219
False Positives	29
False Negatives	8
Accuracy	92.28%
## 5. Code Architecture Overview

Below is a short explanation of how each file functions.

### 🧩 5.1 model.py — SignatureNet CNN

Accepts 400×400 input

Several Conv → ReLU → BatchNorm layers

Outputs 128-dimensional normalized embedding

Shared weights in both Siamese branches

### 🧩 5.2 preprocess.py — Image Standardization

Converts input to a clean 400×400 canvas

Removes noise

Applies binarization

Ensures uniform scale and alignment across samples

### 🧩 5.3 siamese_dataset.py — Positive/Negative Pair Builder

Dynamically generates pairs at runtime

Ensures pos_fraction balance

Applies augmentations:

rotation

translation

scale jitter

brightness/contrast jitter

### 🧩 5.4 siamese_train.py — Training Engine

Handles:

Dataset loading

Augmentation application

Contrastive Loss optimization

Logging distances per epoch

Saving best & last models

Writing siamese_train_metrics.json

Auto-generating training plots

### 🧩 5.5 siamese_evaluate.py — Full Evaluation Pipeline

Loads trained model

Computes pairwise distances

Finds optimal threshold

Outputs metrics & confusion matrix

Generates:

Distribution plots

ROC Curve

PR Curve

# 6. Database Design

The verification system includes a database that manages:

Users

Their multiple signature images

Embeddings extracted by the Siamese model

Mean embedding used for verification

Below is the conceptual design.

### 📘 Entity–Relationship Table Diagram (ASCII Markdown)
+-------------------+
|      USER         |
+-------------------+
| NO (PK)           |
| Name              |
| Surname           |
| MeanEmbedding     |
+---------+---------+
          |
          | 1-to-many
          |
+---------------------------+
|       SIGNATURE           |
+---------------------------+
| SigID (PK)                |
| UserNO (FK → USER.NO)     |
| ImagePath                 |
| EmbeddingVector           |
+---------------------------+

## 7. Supported Database Queries

Here are the required operations described in the project specification.

### 7.1 Query: Verify if a PNG belongs to a user (given NO)

Input:

NO

Signature image (PNG)

Process:

Preprocess image

Generate embedding

Compare with stored user’s MeanEmbedding

If distance < threshold → match

Output:

True/False

Similarity score

### 7.2 Query: Find user’s NO by full name

Input:

Name, Surname
Output:

NO

### 7.3 Query: Identify owner of a given PNG

Input:

Signature image (PNG)
Process:

Compute embedding

Compare with all users’ mean embeddings

Select the closest match below threshold

Output:

Name, Surname, NO

or "No matching user"

### 7.4 Operation: Switch between Siamese & Triplet model

Architecture supports:

Siamese → Contrastive Loss

Triplet → Triplet Loss

A function toggles which embedding model to use.

### 7.5 Operation: Compare two PNG signatures

Given Image A and Image B:

Compute embedding A
Compute embedding B
distance = ||A - B||
Return (distance < threshold)

# 8. Final Remarks

After correcting preprocessing, expanding training, and improving augmentation, the system now:

✔ Achieves 92%+ accuracy
✔ Generalizes to real signatures
✔ Provides clean embedding separation
✔ Supports database-based identity verification
✔ Includes full evaluation and metric logging

This signature verification framework is now production-ready and can be extended with:

Triplet-loss based model comparison

Web API deployment

Database search optimization

GUI integration

