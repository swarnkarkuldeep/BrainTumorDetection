# Brain Tumor Detection

A transfer-learning pipeline that classifies brain MRI scans as **tumor** or **no tumor**,
built on a VGG16 backbone pretrained on ImageNet.

Dataset: [Brain MRI Images for Brain Tumor Detection](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection) (Kaggle).

## Project structure

```
BrainTumorDetection/
├── train.py                     # end-to-end training pipeline
├── predict.py                   # run the trained model on a single image
├── requirements.txt
├── README.md
│
└── (generated after running train.py)
    ├── brain_tumor_detector.keras   # trained model
    ├── label_classes.json           # class name -> index mapping
    ├── plot.jpg                     # training/validation loss & accuracy curves
    ├── confusion_matrix.png         # confusion matrix on the held-out test set
    ├── roc_curve.png                # ROC curve with AUC
    ├── training_history.csv         # per-epoch metrics, both training phases
    └── classification_report.txt    # precision/recall/F1 per class
```

> **Note:** the artifacts under "generated after running train.py" are not included in
> this repo — they're real outputs of an actual training run (a trained model file,
> real metrics, real plots), and only exist once you've run `train.py` yourself against
> the dataset. Committing placeholder versions of them would be misleading, so they're
> left out on purpose. Consider adding them to `.gitignore`, or committing the specific
> run's artifacts once you have them if you want to showcase results in the repo.

## Setup

```bash
pip install -r requirements.txt
```

If you're on Colab, `opencv-python`, `numpy`, `matplotlib`, and `scikit-learn` are usually
preinstalled — you'll mainly need `imutils`.

## Training

```bash
python train.py --dataset-path brain_tumor_dataset
```

If `brain_tumor_dataset/` doesn't exist locally, `train.py` will try to download it from
Kaggle automatically (requires the `kaggle` CLI configured with an API token, or
`kagglehub` installed).

Useful options:

```bash
python train.py \
  --dataset-path brain_tumor_dataset \
  --output-dir . \
  --batch-size 8 \
  --head-epochs 15 \
  --finetune-epochs 10 \
  --head-lr 1e-3 \
  --finetune-lr 1e-5
```

Run `python train.py --help` for the full list of options.

### What training does

1. Loads and preprocesses all images (resize to 224×224, RGB, normalized to [0, 1]).
2. Splits into stratified train / validation / test sets (test set is never touched
   until final evaluation, to avoid leakage).
3. Builds a VGG16 base (frozen, ImageNet weights) with a small classification head
   (`AveragePooling2D → Flatten → Dense(64, relu) → Dropout(0.5) → Dense(softmax)`).
4. **Phase 1:** trains the classification head only, with the VGG16 base frozen.
5. **Phase 2:** unfreezes the last VGG16 conv block and fine-tunes end-to-end at a
   much lower learning rate.
6. Applies class weighting to handle any tumor/no-tumor imbalance in the dataset.
7. Uses early stopping, learning-rate reduction on plateau, and checkpointing
   throughout, so it won't overfit or waste epochs.
8. Evaluates on the held-out test set and saves the confusion matrix, ROC curve,
   classification report, and training curves.

## Predicting on a new image

Once you have a trained model:

```bash
python predict.py --image path/to/scan.jpg
```

```
--- Prediction ---
Image:      scan.jpg
Prediction: yes
Confidence: 96.42%

All class probabilities:
  no        :   3.58%
  yes       :  96.42%
```

By default it looks for `brain_tumor_detector.keras` and `label_classes.json` in the
current directory; override with `--model` and `--labels` if you've moved them.

## Latest run results

Results from an actual training run (`--head-epochs 15 --finetune-epochs 10`, batch size 8):

**Phase 1 — head training (VGG16 frozen, lr = 1e-3)**
Started at 51.7% train accuracy / 76.3% val accuracy (epoch 1) and climbed steadily,
finishing at:

|          | Train | Validation |
| -------- | ----- | ---------- |
| Accuracy | 80.5% | 90.8%      |
| Loss     | 0.421 | 0.306      |

**Phase 2 — fine-tuning (last VGG16 block unfrozen, lr = 1e-5)**
Fine-tuning gave a clear, consistent boost — both accuracy climbing and loss still
dropping every epoch, exactly the pattern you want to see:

| Epoch | Train Acc | Val Acc   | Train Loss | Val Loss  |
| ----- | --------- | --------- | ---------- | --------- |
| 1     | 79.7%     | 88.2%     | 0.440      | 0.275     |
| 3     | 88.4%     | 93.4%     | 0.290      | 0.202     |
| 5     | 92.9%     | 94.7%     | 0.199      | 0.157     |
| 7     | 93.2%     | 96.1%     | 0.167      | 0.123     |
| 10    | **98.0%** | **98.7%** | **0.088**  | **0.058** |

Final epoch: **98.68% validation accuracy, 0.058 validation loss** — no signs of
overfitting (val loss kept falling in step with train loss the whole way through).

> **Note:** the numbers above are _validation_ metrics captured during training. The
> pipeline also runs a final evaluation on a completely held-out **test** set
> afterward (never seen during training or validation) and writes the results to
> `classification_report.txt`, `confusion_matrix.png`, and `roc_curve.png` — those
> are the numbers to cite as the model's true generalization performance. Paste that
> output in and this section can be filled in with exact test accuracy, precision,
> recall, and AUC.

As a rough expectation based on this dataset size (~253 images) and this training
curve, held-out test performance is likely in the 94–98% accuracy range with AUC
above 0.98, but treat that as an estimate until you've got the actual
`classification_report.txt` in hand.

## Disclaimer

This project is for educational purposes only. It is **not** a medical device and must
not be used for actual clinical diagnosis. Any real-world use of MRI-based tumor
detection should involve a qualified radiologist and a properly validated, regulated
system.
