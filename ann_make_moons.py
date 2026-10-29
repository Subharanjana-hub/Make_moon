import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, roc_curve, auc)
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
N_SAMPLES = 1000
NOISE = 0.2
EPOCHS = 200
BATCH_SIZE = 32
OUT_DIR = "ann_moons_outputs"

np.random.seed(SEED)
keras.utils.set_random_seed(SEED)
os.makedirs(OUT_DIR, exist_ok=True)

X, y = make_moons(n_samples=N_SAMPLES, noise=NOISE, random_state=SEED)
print("Dataset shape:", X.shape, "| Labels:", np.unique(y), "| Class counts:", np.bincount(y))

plt.figure(figsize=(6, 5))
plt.scatter(X[y == 0, 0], X[y == 0, 1], c="tab:blue", edgecolor="k", label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], c="tab:orange", edgecolor="k", label="Class 1")
plt.title("make_moons dataset")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/dataset.png", dpi=120)
plt.close()

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.2, random_state=SEED, stratify=y_train_full)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)
print(f"Train: {X_train.shape} | Validation: {X_val.shape} | Test: {X_test.shape}")


def build_ann(input_dim=2):
    model = keras.Sequential([
        layers.Input(shape=(input_dim,)),
        layers.Dense(32, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(16, activation="relu"),
        layers.Dropout(0.1),
        layers.Dense(8, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ], name="ann_make_moons")
    return model


model = build_ann()
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.01),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()

callbacks = [
    keras.callbacks.EarlyStopping(monitor="val_loss", patience=20,
                                  restore_best_weights=True, verbose=1),
    keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                                      patience=8, min_lr=1e-5, verbose=1),
]

history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks,
    verbose=2,
)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(history.history["accuracy"], label="train")
ax[0].plot(history.history["val_accuracy"], label="validation")
ax[0].set_title("Accuracy")
ax[0].set_xlabel("Epoch")
ax[0].legend()
ax[1].plot(history.history["loss"], label="train")
ax[1].plot(history.history["val_loss"], label="validation")
ax[1].set_title("Loss")
ax[1].set_xlabel("Epoch")
ax[1].legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/training_curves.png", dpi=120)
plt.close()

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"\nTest loss    : {test_loss:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

y_prob = model.predict(X_test, verbose=0).ravel()
y_pred = (y_prob >= 0.5).astype(int)
print(f"Accuracy (sklearn): {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification report:\n")
print(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"], digits=4))

cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(4.5, 4))
im = ax.imshow(cm, cmap="Blues")
ax.set_xticks([0, 1])
ax.set_xticklabels(["Class 0", "Class 1"])
ax.set_yticks([0, 1])
ax.set_yticklabels(["Class 0", "Class 1"])
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion Matrix")
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black")
plt.colorbar(im)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/confusion_matrix.png", dpi=120)
plt.close()

fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
plt.figure(figsize=(5, 4.5))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], "k--", alpha=0.5)
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC Curve")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/roc_curve.png", dpi=120)
plt.close()
print(f"ROC AUC: {roc_auc:.4f}")

x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300), np.linspace(y_min, y_max, 300))
grid = np.c_[xx.ravel(), yy.ravel()]
grid_prob = model.predict(scaler.transform(grid), verbose=0).reshape(xx.shape)

plt.figure(figsize=(7, 5.5))
plt.contourf(xx, yy, grid_prob, levels=50, cmap="RdBu", alpha=0.8)
plt.colorbar(label="Probability of Class 1")
plt.contour(xx, yy, grid_prob, levels=[0.5], colors="k", linewidths=2)
X_test_orig = scaler.inverse_transform(X_test)
plt.scatter(X_test_orig[y_test == 0, 0], X_test_orig[y_test == 0, 1],
            c="tab:blue", edgecolor="k", label="Class 0 (test)")
plt.scatter(X_test_orig[y_test == 1, 0], X_test_orig[y_test == 1, 1],
            c="tab:orange", edgecolor="k", label="Class 1 (test)")
plt.title("Decision Boundary learned by the ANN")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/decision_boundary.png", dpi=120)
plt.close()

model.save(f"{OUT_DIR}/ann_moons.keras")
loaded = keras.models.load_model(f"{OUT_DIR}/ann_moons.keras")
new_point = np.array([[0.5, 0.25]])
p = float(loaded.predict(scaler.transform(new_point), verbose=0)[0][0])
print(f"\nNew point {new_point[0].tolist()} -> Class {int(p >= 0.5)} (probability {p:.3f})")
print(f"\nAll outputs saved in: {OUT_DIR}/")
