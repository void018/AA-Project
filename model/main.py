import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
from time import time
from datetime import timedelta

start_time = time()

# files paths
input_file = "main_data.csv"
model_save_file = "model/best_model.pt"
img_file = "model/training_curves.png"

# ─────────────────────────────────────────────
# 1. Load & Feature Engineering
# ─────────────────────────────────────────────
df = pd.read_csv(input_file)

Va = df["V_a"].round(0).values.astype(np.int32)
Vb = df["V_b"].round(0).values.astype(np.int32)
Vc = df["V_c"].round(0).values.astype(np.int32)
Ia = df["phase_a"].round(1).values
Ib = df["phase_b"].round(1).values
Ic = df["phase_c"].round(1).values
y = df["faulted"].values.astype(np.float32)


def engineer_features(Va, Vb, Vc, Ia, Ib, Ic):
    """
    Derives physically meaningful asymmetry features
    from the three post-transformer voltages.
    """
    Va, Vb, Vc = np.abs(Va), np.abs(Vb), np.abs(Vc)
    Ia, Ib, Ic = np.abs(Ia), np.abs(Ib), np.abs(Ic)

    mean_V = (Va + Vb + Vc) / 3.0 + 1e-9  # avoid div-by-zero

    # Pairwise differences (asymmetry indicators)
    diff_ab = np.abs(Va - Vb)
    diff_bc = np.abs(Vb - Vc)
    diff_ca = np.abs(Vc - Va)

    # Deviation from mean (normalized)
    dev_a = (Va - mean_V) / mean_V
    dev_b = (Vb - mean_V) / mean_V
    dev_c = (Vc - mean_V) / mean_V

    # Min/max ratio — a balanced system has this close to 1
    min_V = np.minimum(np.minimum(Va, Vb), Vc)
    max_V = np.maximum(np.maximum(Va, Vb), Vc)
    min_max_ratio = min_V / (max_V + 1e-9)

    # Voltage unbalance factor (NEMA definition)
    max_dev = np.maximum(np.maximum(np.abs(Va-mean_V), np.abs(Vb-mean_V)), np.abs(Vc-mean_V))
    vuf = max_dev / mean_V

    # Negative sequence magnitude (symmetrical components)
    # V2 = (Va + a²·Vb + a·Vc) / 3,  a = e^(j2π/3)
    a = np.exp(1j * 2 * np.pi / 3)
    a2 = np.exp(1j * 4 * np.pi / 3)
    Va_c = Va.astype(complex)
    Vb_c = Vb.astype(complex)
    Vc_c = Vc.astype(complex)
    V2 = np.abs((Va_c + a2 * Vb_c + a * Vc_c) / 3.0)
    V1 = np.abs((Va_c + a * Vb_c + a2 * Vc_c) / 3.0) + 1e-9
    neg_seq_ratio = V2 / V1   # ~0 for healthy, large for fault

    features = np.stack([
        # Va, Vb, Vc,             # raw voltages
        Ia, Ib, Ic,             # raw currents
        # diff_ab, diff_bc, diff_ca,   # pairwise diffs
        dev_a, dev_b, dev_c,    # normalized deviations
        min_max_ratio,          # balance indicator
        vuf,                    # unbalance factor
        # neg_seq_ratio,          # negative sequence (best single indicator)
    ], axis=1)

    return features.astype(np.float32)


X = engineer_features(Va, Vb, Vc, Ia, Ib, Ic)

print(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features")
print(f"Class balance — faulted: {y.mean():.2%}")

# ─────────────────────────────────────────────
# 2. Split & Scale
# ─────────────────────────────────────────────
random_state = 42
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=random_state, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=random_state, stratify=y_temp
)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)


def to_tensors(*arrays):
    return [torch.tensor(a) for a in arrays]


X_tr, y_tr = to_tensors(X_train, y_train)
X_vl, y_vl = to_tensors(X_val,   y_val)
X_te, y_te = to_tensors(X_test,  y_test)

train_loader = DataLoader(TensorDataset(X_tr, y_tr), batch_size=64, shuffle=True)
val_loader = DataLoader(TensorDataset(X_vl, y_vl), batch_size=64)

# ─────────────────────────────────────────────
# 3. Model
# ─────────────────────────────────────────────


class FaultDetector(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            # nn.Linear(64, 32),
            # nn.BatchNorm1d(32),
            # nn.ReLU(),
            # nn.Dropout(0.2),

            nn.Linear(64, 1)   # raw logit — BCEWithLogitsLoss handles sigmoid
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = FaultDetector(X_train.shape[1]).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5)

# ─────────────────────────────────────────────
# 4. Training Loop
# ─────────────────────────────────────────────


def run_epoch(loader, training=True):
    model.train() if training else model.eval()
    total_loss, correct, total = 0.0, 0, 0

    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * len(yb)
            preds = (torch.sigmoid(logits) > 0.5).float()
            correct += (preds == yb).sum().item()
            total += len(yb)

    return total_loss / total, correct / total


history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}
best_val_acc, patience_count, PATIENCE = 0.0, 0, 400

EPOCHS = 400
for epoch in range(1, EPOCHS + 1):
    tr_loss, tr_acc = run_epoch(train_loader, training=True)
    vl_loss, vl_acc = run_epoch(val_loader,   training=False)
    scheduler.step(vl_loss)

    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["train_acc"].append(tr_acc)
    history["val_acc"].append(vl_acc)

    if vl_acc > best_val_acc:
        best_val_acc = vl_acc
        torch.save(model.state_dict(), model_save_file)
        patience_count = 0
    else:
        patience_count += 1

    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d} | Train {tr_acc:.3f} / {tr_loss:.4f} | Val {vl_acc:.3f} / {vl_loss:.4f}")

    if patience_count >= PATIENCE:
        print(f"Early stopping at epoch {epoch}")
        break

# ─────────────────────────────────────────────
# 5. Evaluation on Test Set
# ─────────────────────────────────────────────
# model.load_state_dict(torch.load(model_save_file))
model.eval()

with torch.no_grad():
    logits = model(X_te.to(device))
    preds = (torch.sigmoid(logits) > 0.5).cpu().numpy()

print("\n── Test Set Results ──")
print(classification_report(y_test, preds,
      target_names=["Healthy", "Faulted"]))
print("Confusion Matrix:")
print(confusion_matrix(y_test, preds))
print(f"Traning time: {timedelta(seconds=time() - start_time)}")

# ─────────────────────────────────────────────
# 6. Plots
# ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history["train_loss"], label="Train")
axes[0].plot(history["val_loss"],   label="Val")
axes[0].set_title("Loss")
axes[0].legend()

axes[1].plot(history["train_acc"], label="Train")
axes[1].plot(history["val_acc"],   label="Val")
axes[1].set_title("Accuracy")
axes[1].legend()

plt.tight_layout()
plt.savefig(img_file, dpi=150)
plt.show()


"""
- compare current best model VS calculated nagative sequence
- compare current best model VS the real nagative sequence
"""
