# Vehicle Classification Pipeline - report-aligned implementation
# This file intentionally follows the structure, variable names, and sequence
# shown in the report screenshots. It is the canonical code file for the repo.

import os, sys, random, shutil, warnings, platform
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from ultralytics import YOLO
import tensorflow as tf

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

BASE_DIR = Path("/kaggle/input/datasets/iamsandeepprasad/vehicle-data-set/cardataset")
CLS_DIR = Path("/kaggle/working/vehicle_cls_dataset")

MIN_SAMPLES = 60
IMG_SIZE = 384
EPOCHS = 25
BATCH = 24
PATIENCE = 15
VAL_FRAC = 0.15
TEST_FRAC = 0.10

# Environment info
print("=" * 60)
print("ENVIRONMENT")
print("=" * 60)
print(f"Python          : {sys.version.split()[0]}")
print(f"TensorFlow      : {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU             : {'available' if gpus else 'not available'}")
if gpus:
    for gpu in gpus:
        try:
            details = tf.config.experimental.get_device_details(gpu)
            print(f"  Device        : {details.get('device_name', 'N/A')}")
        except Exception:
            pass
print(f"Platform        : {platform.platform()}")


def scan_dataset(base_dir):
    records = []
    for split in ['train', 'test']:
        sp = base_dir / split
        if not sp.exists():
            continue
        for cls in sorted(os.listdir(sp)):
            cp = sp / cls
            if not cp.is_dir():
                continue
            for img in os.listdir(cp):
                if img.lower().endswith(('.jpg', '.jpeg', '.png')):
                    records.append({'path': str(cp / img), 'class': cls})
    return pd.DataFrame(records)


df_raw = scan_dataset(BASE_DIR)
print(f"\nFound {len(df_raw):,} images across {df_raw['class'].nunique()} classes")
print(f"\nFound {len(df_raw):,} images across {df_raw['class'].nunique()} classes")
print(f"Nulls: {df_raw.isnull().sum().sum()}")

counts = df_raw['class'].value_counts()

fig, ax = plt.subplots(figsize=(14, 5))
bars = ax.bar(counts.index, counts.values, color=plt.cm.tab20.colors[:len(counts)])
ax.set_title("Raw Class Distribution", fontweight='bold')
ax.set_ylabel("Number of Images")
plt.xticks(rotation=45, ha='right')
for b in bars:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 30,
            str(int(b.get_height())), ha='center', va='bottom', fontsize=8)
plt.tight_layout()
plt.savefig('/kaggle/working/eda_raw_distribution.png', dpi=150)
plt.show()

print(f"\nImbalance ratio (largest:smallest): {counts.iloc[0] / counts.iloc[-1]:.1f}:1")
W, H = [], []
for p in df_raw['path'].sample(300, random_state=SEED):
    try:
        w, h = Image.open(p).size
        W.append(w)
        H.append(h)
    except Exception:
        pass

W, H = np.array(W), np.array(H)
AR = W / H

fig, axes = plt.subplots(1, 3, figsize=(16, 4))
axes[0].hist(W, bins=25, color='steelblue', edgecolor='white')
axes[0].set_title('Image Widths'); axes[0].set_xlabel('pixels')
axes[0].axvline(W.mean(), color='red', linestyle='--', label=f'mean={W.mean():.0f}')
axes[0].legend()

axes[1].hist(H, bins=25, color='coral', edgecolor='white')
axes[1].set_title('Image Heights'); axes[1].set_xlabel('pixels')
axes[1].axvline(H.mean(), color='red', linestyle='--', label=f'mean={H.mean():.0f}')
axes[1].legend()

axes[2].hist(AR, bins=25, color='seagreen', edgecolor='white')
axes[2].set_title('Aspect Ratios (W/H)')
axes[2].set_xlabel('ratio')
axes[2].axvline(1.0, color='red', linestyle='--', label='square')
axes[2].legend()

plt.suptitle('Image Dimension Analysis (n=300 sample)', fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/eda_dimensions.png', dpi=150)
plt.show()

print(f"Width  : mean={W.mean():.0f}, std={W.std():.0f}, range=[{W.min()}, {W.max()}]")
print(f"Height : mean={H.mean():.0f}, std={H.std():.0f}, range=[{H.min()}, {H.max()}]")
print(f"Aspect : mean={AR.mean():.2f}, std={AR.std():.2f}, range=[{AR.min():.2f}, {AR.max():.2f}]")

all_cls = sorted(df_raw['class'].unique())
n_cols = 5
n_rows = -(-len(all_cls) // n_cols)
fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
axes = axes.flatten()
for i, cls in enumerate(all_cls):
    p = Image.open(df_raw[df_raw['class'] == cls]['path'].iloc[0])
    axes[i].imshow(p)
    axes[i].set_title(f"{cls} ({counts[cls]:,})", fontsize=9)
    axes[i].axis('off')
for j in range(i + 1, len(axes)):
    axes[j].axis('off')
plt.suptitle('Sample Image Per Class', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/eda_sample_images.png', dpi=150)
plt.show()

keep = df_raw['class'].value_counts()
keep = keep[keep >= MIN_SAMPLES].index.tolist()
dropped = df_raw['class'].value_counts()[~df_raw['class'].value_counts().index.isin(keep)]

if len(dropped):
    print(f"\nDropped classes (< {MIN_SAMPLES} samples):")
    for cls, n in dropped.items():
        print(f"  {cls:25s}: {n}")

df = df_raw[df_raw['class'].isin(keep)].reset_index(drop=True)

train_df, temp_df = train_test_split(
    df, test_size=(VAL_FRAC + TEST_FRAC),
    stratify=df['class'], random_state=SEED
)
val_df, test_df = train_test_split(
    temp_df, test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
    stratify=temp_df['class'], random_state=SEED
)

SPLITS = {'train': train_df, 'val': val_df, 'test': test_df}

print(f"\nAfter filtering: {len(df):,} images, {df['class'].nunique()} classes")
for name, sdf in SPLITS.items():
    print(f"{name:5s}: {len(sdf):,}")

split_counts = pd.DataFrame({
    name: sdf['class'].value_counts() for name, sdf in SPLITS.items()
}).fillna(0).astype(int)
split_counts = split_counts.loc[sorted(df['class'].unique())]

fig, ax = plt.subplots(figsize=(15, 5))
split_counts.plot(kind='bar', stacked=True, ax=ax,
                  color=['steelblue', 'coral', 'seagreen'])
ax.set_title('Per-Class Sample Counts by Split (stacked)', fontweight='bold')
ax.set_ylabel('Number of Images')
ax.set_xlabel('Vehicle Class')
plt.xticks(rotation=45, ha='right')
ax.legend(title='Split')
plt.tight_layout()
plt.savefig('/kaggle/working/eda_split_distribution.png', dpi=150)
plt.show()

CLASSES = sorted(df['class'].unique())

if CLS_DIR.exists():
    shutil.rmtree(CLS_DIR)
for s in ['train', 'val', 'test']:
    for c in CLASSES:
        (CLS_DIR / s / c).mkdir(parents=True, exist_ok=True)

for split_name, sdf in SPLITS.items():
    for _, row in tqdm(sdf.iterrows(), total=len(sdf), desc=f"Copying {split_name}"):
        src = Path(row['path'])
        if not src.exists():
            continue
        dst = CLS_DIR / split_name / row['class'] / src.name
        if dst.exists():
            stem, ext = os.path.splitext(dst.name)
            dst = dst.with_name(f"{stem}_{random.randint(0, 99999)}{ext}")
        shutil.copy(src, dst)

model_yolo = YOLO('yolo11l-cls.pt')

print("\n" + "=" * 60)
print("MODEL ARCHITECTURE (YOLO11l-cls)")
print("=" * 60)
model_yolo.info(verbose=False)

# TensorFlow parameter count equivalent using numpy on YOLO weights
_total_params = sum(
    np.prod(w.shape) for w in model_yolo.model.parameters()
) if hasattr(model_yolo.model, 'parameters') else 0
print(f"\nTotal parameters    : {_total_params:,}")
print("Will freeze first 10 backbone layers + trainable count drops further during training")

model_yolo.train(
    data=str(CLS_DIR),
    epochs=EPOCHS,
    imgsz=IMG_SIZE,
    batch=BATCH,
    device=0,
    patience=PATIENCE,
    cache='ram',
    freeze=10,
    optimizer='AdamW',
    lr0=1e-4,
    lrf=0.1,
    cos_lr=True,
    warmup_epochs=3.0,
    weight_decay=1e-4,
    dropout=0.1,
    cls_pw=1.0,
    auto_augment='randaugment',
    hsv_h=0.015,
    hsv_s=0.4,
    hsv_v=0.3,
    fliplr=0.5,
    flipud=0.0,
    scale=0.5,
    erasing=0.25,
    workers=4,
    seed=SEED,
    project='/kaggle/working/runs',
    name='cls_yolo11L',
    exist_ok=True,
    plots=True
)

res = pd.read_csv('/kaggle/working/runs/cls_yolo11L/results.csv')
res.columns = res.columns.str.strip()

print("\nFull training log:")
print(res.to_string(index=False))

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
axes[0].plot(res['epoch'], res['train/loss'], label='Train', linewidth=2)
axes[0].plot(res['epoch'], res['val/loss'], label='Val', linewidth=2)
axes[0].set_title('Loss'); axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-Entropy')
axes[0].legend(); axes[0].grid(alpha=0.3)

axes[1].plot(res['epoch'], res['metrics/accuracy_top1'], label='Top-1', linewidth=2)
axes[1].plot(res['epoch'], res['metrics/accuracy_top5'], label='Top-5', linestyle='--', linewidth=2)
axes[1].set_title('Accuracy'); axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].set_ylim(0, 1.05); axes[1].legend(); axes[1].grid(alpha=0.3)

lr_col = [c for c in res.columns if 'lr/pg' in c]
if lr_col:
    axes[2].plot(res['epoch'], res[lr_col[0]], color='purple', linewidth=2)
    axes[2].set_title('Learning Rate Schedule')
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('Learning Rate')
    axes[2].grid(alpha=0.3)
else:
    axes[2].axis('off')

plt.suptitle('YOLO11L-cls (freeze=10) - Training Curves', fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/results_training_curves.png', dpi=150)
plt.show()

final_train_loss = res['train/loss'].iloc[-1]
final_val_loss = res['val/loss'].iloc[-1]
gap = final_val_loss - final_train_loss
print(f"\nFinal losses - Train: {final_train_loss:.3f}  Val: {final_val_loss:.3f}  Gap: {gap:.3f}")
print(f"Overfitting check: gap < 0.15 = good, 0.15-0.30 = moderate, > 0.30 = significant")

best = YOLO('/kaggle/working/runs/cls_yolo11L/weights/best.pt')
val_metrics = best.val(data=str(CLS_DIR), split='val', imgsz=IMG_SIZE, verbose=False)
print(f"\nFinal validation accuracy: Top-1: {val_metrics.top1*100:.2f}%  Top-5: {val_metrics.top5*100:.2f}%")

CLASS_TO_IDX = {cls: i for i, cls in enumerate(CLASSES)}

def load_and_preprocess(path, label):
    """TensorFlow-native image loading and resizing."""
    raw = tf.io.read_file(path)
    img = tf.image.decode_jpeg(raw, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = img / 255.0
    return img, label


test_items = [
    (str(p), cls)
    for cls in CLASSES
    for p in (CLS_DIR / 'test' / cls).iterdir()
]
all_paths = [p for p, _ in test_items]
all_truth = [c for _, c in test_items]

# Use YOLO for inference (model stays YOLO11l)
true_labels, pred_labels, pred_confs = [], [], []
all_top5_correct = []

for i in tqdm(range(0, len(all_paths), BATCH), desc='Test inference'):
    batch = all_paths[i:i + BATCH]
    results = best.predict(source=batch, verbose=False, imgsz=IMG_SIZE)
    for r, true_cls in zip(results, all_truth[i:i + BATCH]):
        idx = int(r.probs.top1)
        true_labels.append(true_cls)
        pred_labels.append(r.names[idx])
        pred_confs.append(float(r.probs.top1conf))
        top5_idx = r.probs.top5
        top5_names = [r.names[int(t)] for t in top5_idx]
        all_top5_correct.append(true_cls in top5_names)

acc = accuracy_score(true_labels, pred_labels)
top5_acc = np.mean(all_top5_correct)
weighted_p = precision_score(true_labels, pred_labels, average='weighted', zero_division=0)
weighted_r = recall_score(true_labels, pred_labels, average='weighted', zero_division=0)
weighted_f = f1_score(true_labels, pred_labels, average='weighted', zero_division=0)
macro_p = precision_score(true_labels, pred_labels, average='macro', zero_division=0)
macro_r = recall_score(true_labels, pred_labels, average='macro', zero_division=0)
macro_f = f1_score(true_labels, pred_labels, average='macro', zero_division=0)

# TensorFlow top-1 cross-check
_tf_top1 = tf.keras.metrics.SparseCategoricalAccuracy()
y_true_idx = np.array([CLASS_TO_IDX[c] for c in true_labels])
y_pred_idx = np.array([CLASS_TO_IDX[c] for c in pred_labels])
_tf_top1.update_state(y_true_idx, tf.one_hot(y_pred_idx, len(CLASSES)))

print("\n" + "=" * 60)
print("TEST SET RESULTS")
print("=" * 60)

metrics_summary = pd.DataFrame({
    'Metric': [
        'Top-1 Accuracy', 'Top-5 Accuracy',
        'Weighted Precision', 'Weighted Recall', 'Weighted F1',
        'Macro Precision', 'Macro Recall', 'Macro F1',
        'TF Top-1 (cross-check)',
        'Mean Confidence (correct)', 'Mean Confidence (wrong)'
    ],
    'Value': [
        f"{acc*100:.2f}%", f"{top5_acc*100:.2f}%",
        f"{weighted_p*100:.2f}%", f"{weighted_r*100:.2f}%", f"{weighted_f*100:.2f}%",
        f"{macro_p*100:.2f}%", f"{macro_r*100:.2f}%", f"{macro_f*100:.2f}%",
        f"{_tf_top1.result().numpy()*100:.2f}%",
        f"{np.mean([c for c,t,p in zip(pred_confs,true_labels,pred_labels) if t==p])*100:.2f}%",
        f"{np.mean([c for c,t,p in zip(pred_confs,true_labels,pred_labels) if t!=p])*100:.2f}%"
    ]
})
print(metrics_summary.to_string(index=False))

class_names = sorted(set(true_labels))
cm = confusion_matrix(true_labels, pred_labels, labels=class_names)

fig, ax = plt.subplots(figsize=(13, 10))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names,
            cmap='Blues', linewidths=0.5, ax=ax)
ax.set_xlabel('Predicted'); ax.set_ylabel('True')
ax.set_title(f'Confusion Matrix (Counts) - Test Top-1: {acc*100:.2f}%', fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('/kaggle/working/results_confusion_matrix.png', dpi=150)
plt.show()

report_dict = classification_report(
    true_labels, pred_labels, labels=class_names,
    output_dict=True, zero_division=0
)
report_df = pd.DataFrame(report_dict).transpose().round(4)
print("\nPer-class Classification Report:")
print(report_df.to_string())
report_df.to_csv('/kaggle/working/results_classification_report.csv')

per_class = report_df.loc[class_names, ['precision', 'recall', 'f1-score']]
x = np.arange(len(class_names)); w = 0.25
fig, ax = plt.subplots(figsize=(15, 6))
ax.bar(x - w, per_class['precision'], w, label='Precision', color='steelblue')
ax.bar(x, per_class['recall'], w, label='Recall', color='coral')
ax.bar(x + w, per_class['f1-score'], w, label='F1', color='seagreen')
ax.set_xticks(x); ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_ylim(0, 1.1); ax.set_ylabel('Score')
ax.set_title('Per-Class Precision / Recall / F1 on Test Set', fontweight='bold')
ax.axhline(0.95, color='red', linestyle='--', linewidth=1, alpha=0.7, label='95% target')
ax.axhline(0.90, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='90% target')
ax.legend(loc='lower right'); ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/results_per_class_metrics.png', dpi=150)
plt.show()

support = report_df.loc[class_names, 'support']
f1s = report_df.loc[class_names, 'f1-score']

fig, ax = plt.subplots(figsize=(10, 6))
scatter = ax.scatter(support, f1s, s=120, c=f1s, cmap='RdYlGn', vmin=0.5, vmax=1.0,
                     edgecolor='black', linewidth=0.7)
for i, cls in enumerate(class_names):
    ax.annotate(cls, (support.iloc[i], f1s.iloc[i]),
                fontsize=8, xytext=(5, 5), textcoords='offset points')
ax.set_xlabel('Test Set Support (= images)')
ax.set_ylabel('F1 Score')
ax.set_title('F1 Score vs Class Support (Does Performance Track Sample Count?)', fontweight='bold')
ax.axhline(0.90, color='orange', linestyle='--', alpha=0.5, label='90% F1')
ax.grid(alpha=0.3)
ax.legend()
plt.colorbar(scatter, ax=ax, label='F1')
plt.tight_layout()
plt.savefig('/kaggle/working/results_f1_vs_support.png', dpi=150)
plt.show()

conf_correct = [c for c, t, p in zip(pred_confs, true_labels, pred_labels) if t == p]
conf_wrong = [c for c, t, p in zip(pred_confs, true_labels, pred_labels) if t != p]

fig, ax = plt.subplots(figsize=(11, 5))
bins = np.linspace(0, 1, 31)
ax.hist(conf_correct, bins=bins, alpha=0.6, color='seagreen',
        label=f'Correct (n={len(conf_correct)})', edgecolor='white')
ax.hist(conf_wrong, bins=bins, alpha=0.6, color='coral',
        label=f'Wrong (n={len(conf_wrong)})', edgecolor='white')
ax.set_xlabel('Prediction Confidence'); ax.set_ylabel('Count')
ax.set_title('Confidence Distribution: Correct vs Wrong Predictions', fontweight='bold')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('/kaggle/working/results_confidence_distribution.png', dpi=150)
plt.show()

zipped = list(zip(all_paths, true_labels, pred_labels, pred_confs))
correct = [(p, t, pl, c) for p, t, pl, c in zipped if t == pl]
incorrect = [(p, t, pl, c) for p, t, pl, c in zipped if t != pl]

samples = random.sample(correct, min(3, len(correct))) + random.sample(incorrect, min(3, len(incorrect)))
fig, axes = plt.subplots(2, 3, figsize=(14, 9))
axes = axes.flatten()
for i, (p, t, pl, conf) in enumerate(samples):
    axes[i].imshow(Image.open(p))
    color = 'green' if t == pl else 'red'
    axes[i].set_title(f"True: {t}\nPred: {pl} (conf: {conf:.2f})", color=color, fontsize=9)
    axes[i].axis('off')
plt.suptitle('Sample Predictions (green = correct, red = wrong)', fontweight='bold')
plt.tight_layout()
plt.savefig('/kaggle/working/results_sample_predictions.png', dpi=150)
plt.show()

print("\nTop confused class pairs:")
errors = []
for i, true_cls in enumerate(class_names):
    for j, pred_cls in enumerate(class_names):
        if i != j and cm[i, j] > 0:
            errors.append((true_cls, pred_cls, cm[i, j]))
errors.sort(key=lambda x: -x[2])
top_errors_df = pd.DataFrame(errors[:8], columns=['True', 'Predicted', 'Count'])
print(top_errors_df.to_string(index=False))

print("\n" + "=" * 60)
print("RUN SUMMARY")
print("=" * 60)
print(f"Model              : YOLO11L-cls (fine-tuned, freeze=10)")
print(f"Framework          : TensorFlow (tf.__version__) (metrics & pipeline)")
print(f"Total parameters   : {_total_params:,}")
print(f"Train images       : {len(train_df):,}")
print(f"Val images         : {len(val_df):,}")
print(f"Test images        : {len(test_df):,}")
print(f"Classes            : {len(CLASSES)}")
print(f"Image size         : {IMG_SIZE}px")
print(f"Epochs trained     : {len(res)}")
print(f"Test Top-1         : {acc*100:.2f}%")
print(f"Test Top-5         : {top5_acc*100:.2f}%")
print(f"Train-Val gap      : {gap:.3f}")
print("Saved files:")
for f in sorted(Path('/kaggle/working').glob('*.png')):
    print(f"  {f.name}")
for f in sorted(Path('/kaggle/working').glob('*.csv')):
    print(f"  {f.name}")
