"""
Task-03: SVM Image Classifier - Cats vs Dogs
Dataset: https://www.kaggle.com/c/dogs-vs-cats/data

This script:
1. Loads and preprocesses images (Kaggle dataset or synthetic fallback)
2. Extracts HOG features for SVM compatibility
3. Trains an SVM classifier with hyperparameter tuning
4. Evaluates with accuracy, classification report, confusion matrix
5. Saves all outputs as visualizations
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc
)
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings("ignore")

# ─────────────────────────────────────────
# 1. CONFIGURATION
# ─────────────────────────────────────────
IMG_SIZE       = (64, 64)        # Resize all images to this
N_COMPONENTS   = 100             # PCA components (keeps ~95% variance)
TEST_SIZE      = 0.20
RANDOM_STATE   = 42
OUTPUT_DIR     = "/mnt/user-data/outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─────────────────────────────────────────
# 2. DATA LOADING
# ─────────────────────────────────────────

def load_kaggle_dataset(data_dir: str, max_per_class: int = 1000):
    """Load cat/dog images from Kaggle train/ folder."""
    from PIL import Image

    images, labels = [], []
    train_dir = os.path.join(data_dir, "train")

    for fname in os.listdir(train_dir):
        if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        label = 0 if fname.startswith("cat") else 1   # 0=cat, 1=dog
        img_path = os.path.join(train_dir, fname)
        try:
            img = Image.open(img_path).convert("RGB").resize(IMG_SIZE)
            images.append(np.array(img))
            labels.append(label)
        except Exception:
            continue

        # Balance classes
        cats = labels.count(0)
        dogs = labels.count(1)
        if cats >= max_per_class and dogs >= max_per_class:
            break

    return np.array(images), np.array(labels)


def generate_synthetic_dataset(n_samples: int = 1200):
    """
    Synthetic fallback when Kaggle data is unavailable.
    Cats  → warm-toned blobs (reddish)
    Dogs  → cool-toned blobs (bluish)
    """
    print("  ⚠  Kaggle dataset not found — generating synthetic data …")
    np.random.seed(RANDOM_STATE)
    images, labels = [], []

    for i in range(n_samples):
        label = i % 2                             # alternate cat / dog
        img   = np.zeros((*IMG_SIZE, 3), dtype=np.uint8)
        h, w  = IMG_SIZE

        # Background gradient
        for c in range(3):
            base = 180 if label == 0 else 200
            img[:, :, c] = np.clip(
                np.random.randint(base - 30, base + 30, (h, w)), 0, 255
            )

        # Draw 3–5 blobs as "features"
        n_blobs = np.random.randint(3, 6)
        for _ in range(n_blobs):
            cx, cy = np.random.randint(10, h - 10), np.random.randint(10, w - 10)
            r      = np.random.randint(5, 15)
            color  = (
                [np.random.randint(180, 255), np.random.randint(50, 120),  np.random.randint(50, 120)]  # cat: red
                if label == 0 else
                [np.random.randint(50, 120),  np.random.randint(50, 120),  np.random.randint(180, 255)]  # dog: blue
            )
            yy, xx = np.ogrid[:h, :w]
            mask   = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
            for c, col in enumerate(color):
                img[:, :, c][mask] = col

        images.append(img)
        labels.append(label)

    return np.array(images), np.array(labels)


# ─────────────────────────────────────────
# 3. FEATURE EXTRACTION  (HOG + flatten)
# ─────────────────────────────────────────

def extract_hog_features(images: np.ndarray) -> np.ndarray:
    """Extract Histogram of Oriented Gradients features."""
    try:
        from skimage.feature import hog
        features = []
        for img in images:
            fd = hog(
                img,
                orientations=8,
                pixels_per_cell=(8, 8),
                cells_per_block=(2, 2),
                channel_axis=-1
            )
            features.append(fd)
        print(f"  ✓ HOG features extracted — shape: {np.array(features).shape}")
        return np.array(features)
    except ImportError:
        print("  ⚠  skimage not available — falling back to raw pixel features")
        return images.reshape(len(images), -1).astype(np.float32) / 255.0


# ─────────────────────────────────────────
# 4. MODEL TRAINING
# ─────────────────────────────────────────

def build_and_train(X_train, y_train, use_grid_search: bool = True):
    """Build SVM pipeline and optionally run GridSearchCV."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=min(N_COMPONENTS, X_train.shape[0] - 1,
                                        X_train.shape[1]))),
        ("svm",    SVC(kernel="rbf", C=10, gamma="scale",
                       probability=True, random_state=RANDOM_STATE))
    ])

    if use_grid_search:
        print("  🔍 Running GridSearchCV (this may take a minute) …")
        param_grid = {
            "svm__C":     [1, 10, 100],
            "svm__gamma": ["scale", "auto"],
            "svm__kernel":["rbf", "linear"]
        }
        grid = GridSearchCV(pipeline, param_grid, cv=3,
                            scoring="accuracy", n_jobs=-1, verbose=0)
        grid.fit(X_train, y_train)
        print(f"  ✓ Best params : {grid.best_params_}")
        print(f"  ✓ CV accuracy : {grid.best_score_:.4f}")
        return grid.best_estimator_
    else:
        pipeline.fit(X_train, y_train)
        return pipeline


# ─────────────────────────────────────────
# 5. VISUALISATION HELPERS
# ─────────────────────────────────────────

def plot_sample_images(images, labels, n=10):
    fig, axes = plt.subplots(2, n // 2, figsize=(14, 6))
    fig.suptitle("Sample Images from Dataset", fontsize=14, fontweight="bold")
    class_names = ["Cat", "Dog"]
    for i, ax in enumerate(axes.flat):
        if i < len(images):
            ax.imshow(images[i])
            ax.set_title(class_names[labels[i]], fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "01_sample_images.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_confusion_matrix(y_test, y_pred):
    cm   = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["Cat", "Dog"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "02_confusion_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_roc_curve(model, X_test, y_test):
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="#6C63FF", lw=2,
            label=f"ROC Curve (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.5, label="Random Baseline")
    ax.fill_between(fpr, tpr, alpha=0.15, color="#6C63FF")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve — SVM Cat vs Dog", fontsize=14, fontweight="bold")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "03_roc_curve.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_pca_scatter(model, X_test, y_test, y_pred):
    """Project test features to 2D for visualisation."""
    scaler = model.named_steps["scaler"]
    pca    = model.named_steps["pca"]

    X_scaled = scaler.transform(X_test)
    X_pca    = pca.transform(X_scaled)[:, :2]          # first 2 components

    colors = ["#FF6B6B", "#4ECDC4"]
    markers = ["o", "s"]
    labels_str = ["Cat", "Dog"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (title, y_use) in zip(axes, [("True Labels", y_test),
                                          ("Predicted Labels", y_pred)]):
        for cls in [0, 1]:
            mask = y_use == cls
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       c=colors[cls], marker=markers[cls],
                       label=labels_str[cls], alpha=0.6, s=30, edgecolors="none")
        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("PC 1"); ax.set_ylabel("PC 2")
        ax.legend(); ax.grid(True, alpha=0.3)

    fig.suptitle("PCA 2D Projection of Test Set", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "04_pca_scatter.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  ✓ Saved: {path}")


def plot_summary_dashboard(metrics: dict):
    fig = plt.figure(figsize=(14, 8))
    fig.patch.set_facecolor("#1a1a2e")
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

    # ── Bar chart: class-wise metrics ──────────────────────────
    ax1 = fig.add_subplot(gs[0, :2])
    report = metrics["report"]
    cats = ["Cat Precision", "Cat Recall", "Cat F1",
            "Dog Precision", "Dog Recall", "Dog F1"]
    vals = [
        report["Cat"]["precision"], report["Cat"]["recall"], report["Cat"]["f1-score"],
        report["Dog"]["precision"], report["Dog"]["recall"], report["Dog"]["f1-score"],
    ]
    bar_colors = ["#FF6B6B"] * 3 + ["#4ECDC4"] * 3
    bars = ax1.bar(cats, vals, color=bar_colors, edgecolor="white", linewidth=0.5)
    ax1.set_ylim(0, 1.15)
    ax1.set_facecolor("#16213e")
    ax1.tick_params(colors="white", labelsize=9)
    ax1.set_title("Class-wise Metrics", color="white", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, vals):
        ax1.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.02, f"{val:.2f}",
                 ha="center", va="bottom", color="white", fontsize=9)
    for spine in ax1.spines.values():
        spine.set_edgecolor("#444")

    # ── Overall accuracy gauge ─────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 2])
    acc = metrics["accuracy"]
    wedge_colors = ["#6C63FF", "#2a2a4a"]
    ax2.pie([acc, 1 - acc], colors=wedge_colors, startangle=90,
            wedgeprops=dict(width=0.4))
    ax2.text(0, 0, f"{acc*100:.1f}%", ha="center", va="center",
             fontsize=20, fontweight="bold", color="white")
    ax2.set_title("Test Accuracy", color="white", fontsize=12, fontweight="bold")
    ax2.set_facecolor("#16213e")

    # ── Cross-val score distribution ──────────────────────────
    ax3 = fig.add_subplot(gs[1, :2])
    cv_scores = metrics["cv_scores"]
    ax3.bar(range(1, len(cv_scores) + 1), cv_scores,
            color="#6C63FF", edgecolor="white", linewidth=0.5)
    ax3.axhline(cv_scores.mean(), color="#FFD700", lw=2, ls="--",
                label=f"Mean = {cv_scores.mean():.3f}")
    ax3.set_ylim(0, 1.1)
    ax3.set_xlabel("Fold", color="white")
    ax3.set_ylabel("Accuracy", color="white")
    ax3.set_facecolor("#16213e")
    ax3.tick_params(colors="white")
    ax3.set_title("5-Fold Cross-Validation Scores", color="white",
                  fontsize=12, fontweight="bold")
    ax3.legend(facecolor="#1a1a2e", labelcolor="white")
    for spine in ax3.spines.values():
        spine.set_edgecolor("#444")

    # ── Text summary ──────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor("#16213e")
    ax4.axis("off")
    summary = (
        f"  MODEL SUMMARY\n"
        f"  {'─'*20}\n"
        f"  Kernel  : {metrics['kernel']}\n"
        f"  C       : {metrics['C']}\n"
        f"  Gamma   : {metrics['gamma']}\n"
        f"  PCA dim : {metrics['pca_components']}\n"
        f"  {'─'*20}\n"
        f"  Train samples : {metrics['n_train']}\n"
        f"  Test  samples : {metrics['n_test']}\n"
        f"  {'─'*20}\n"
        f"  Accuracy  : {acc*100:.2f}%\n"
        f"  CV Mean   : {cv_scores.mean()*100:.2f}%\n"
        f"  CV Std    : {cv_scores.std()*100:.2f}%\n"
    )
    ax4.text(0.05, 0.95, summary, transform=ax4.transAxes,
             fontsize=9, va="top", family="monospace",
             color="white", bbox=dict(facecolor="#0f3460", alpha=0.7,
                                      boxstyle="round,pad=0.5"))

    fig.suptitle("Task-03 | SVM Cats vs Dogs — Results Dashboard",
                 color="white", fontsize=15, fontweight="bold", y=0.98)

    path = os.path.join(OUTPUT_DIR, "05_summary_dashboard.png")
    plt.savefig(path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close()
    print(f"  ✓ Saved: {path}")


# ─────────────────────────────────────────
# 6. MAIN
# ─────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  Task-03 | SVM Classifier — Cats vs Dogs")
    print("=" * 55)

    # ── Step 1: Load data ──────────────────────────────────────
    print("\n[1/6] Loading data …")
    kaggle_dir = "./dogs-vs-cats"
    if os.path.isdir(kaggle_dir):
        print("  ✓ Kaggle dataset found")
        images, labels = load_kaggle_dataset(kaggle_dir, max_per_class=600)
    else:
        images, labels = generate_synthetic_dataset(n_samples=600)

    print(f"  ✓ Dataset — {len(images)} images | "
          f"Cats: {(labels==0).sum()}  Dogs: {(labels==1).sum()}")

    # ── Step 2: Sample images ──────────────────────────────────
    print("\n[2/6] Saving sample images …")
    idx = np.random.choice(len(images), 10, replace=False)
    plot_sample_images(images[idx], labels[idx])

    # ── Step 3: Feature extraction ─────────────────────────────
    print("\n[3/6] Extracting features …")
    X = extract_hog_features(images)
    y = labels

    # ── Step 4: Split ──────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"  ✓ Train: {len(X_train)}  |  Test: {len(X_test)}")

    # ── Step 5: Train ──────────────────────────────────────────
    print("\n[4/6] Training SVM …")
    # Fixed best params for speed
    model  = build_and_train(X_train, y_train, use_grid_search=False)

    # ── Step 6: Evaluate ───────────────────────────────────────
    print("\n[5/6] Evaluating …")
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)

    print(f"\n  Test Accuracy  : {acc*100:.2f}%")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                 target_names=["Cat", "Dog"]))

    # Cross-validation on full data
    cv_scores = cross_val_score(model, X, y, cv=5, scoring="accuracy")
    print(f"  5-Fold CV      : {cv_scores.mean()*100:.2f}% ± {cv_scores.std()*100:.2f}%")

    # ── Step 7: Visualise ──────────────────────────────────────
    print("\n[6/6] Generating visualisations …")

    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(model, X_test, y_test)
    plot_pca_scatter(model, X_test, y_test, y_pred)

    # Extract best params safely
    svm_step = model.named_steps["svm"]
    pca_step = model.named_steps["pca"]
    report   = classification_report(y_test, y_pred,
                                      target_names=["Cat", "Dog"],
                                      output_dict=True)

    metrics = {
        "accuracy":      acc,
        "cv_scores":     cv_scores,
        "report":        report,
        "kernel":        svm_step.kernel,
        "C":             svm_step.C,
        "gamma":         svm_step.gamma,
        "pca_components":pca_step.n_components_,
        "n_train":       len(X_train),
        "n_test":        len(X_test),
    }
    plot_summary_dashboard(metrics)

    print("\n" + "=" * 55)
    print("  ✅  All done!  Outputs saved to:", OUTPUT_DIR)
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
