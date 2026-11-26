
"""
EyeSense: Contrastive Learning + Bi-GRU
---------------------------------------
Phase 1: Self-supervised contrastive pretraining
Phase 2: Supervised classification fine-tuning
Phase 3: Final test evaluation

Group 1 = Dyslexic
Group 2 = Non-dyslexic

Input features:
Word_Number, FIX_X, FIX_Y, FIX_DURATION
"""

from pathlib import Path
import random
import copy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


# ---------------------------------------------------------
# 1. CONFIGURATION
# ---------------------------------------------------------

SEED = 42
BATCH_SIZE = 64
CONTRASTIVE_EPOCHS = 20
CLASSIFICATION_EPOCHS = 40
LEARNING_RATE = 0.001
TEMPERATURE = 0.2
PATIENCE = 8

INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 2
PROJECTION_SIZE = 32

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_FOLDER = PROJECT_ROOT / "data" / "processed" / "splits"
MODEL_FOLDER = PROJECT_ROOT / "models" / "saved"

MODEL_FOLDER.mkdir(parents=True, exist_ok=True)

TRAIN_PATH = DATA_FOLDER / "train_data.npz"
VAL_PATH = DATA_FOLDER / "validation_data.npz"
TEST_PATH = DATA_FOLDER / "test_data.npz"

OUTPUT_PATH = MODEL_FOLDER / "contrastive_bigru_best.pt"


# ---------------------------------------------------------
# 2. REPRODUCIBILITY
# ---------------------------------------------------------

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------
# 3. LOAD PARTICIPANT SEQUENCES
# ---------------------------------------------------------

def load_split(path):

    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    data = np.load(path, allow_pickle=True)

    sequences = list(data["sequences"])
    groups = np.asarray(data["groups"], dtype=np.int64)

    if len(sequences) != len(groups):
        raise ValueError("Sequence and label counts do not match.")

    processed_sequences = []

    for sequence in sequences:

        sequence = np.asarray(sequence, dtype=np.float32).copy()

        if sequence.ndim != 2 or sequence.shape[1] != INPUT_SIZE:
            raise ValueError(
                f"Unexpected sequence shape: {sequence.shape}"
            )

        if not np.isfinite(sequence).all():
            raise ValueError("Non-finite values found in sequence.")

        if np.any(sequence[:, 3] < 0):
            raise ValueError("Negative fixation duration found.")

        # Log transformation reduces the influence of very
        # large fixation-duration values.
        sequence[:, 3] = np.log1p(sequence[:, 3])

        processed_sequences.append(sequence)

    # Group 1 -> class 0 (Dyslexic)
    # Group 2 -> class 1 (Non-dyslexic)
    labels = groups - 1

    return processed_sequences, labels


# ---------------------------------------------------------
# 4. NORMALIZATION USING TRAINING DATA ONLY
# ---------------------------------------------------------

def calculate_normalization(training_sequences):

    all_frames = np.concatenate(training_sequences, axis=0)

    mean = all_frames.mean(axis=0)
    std = all_frames.std(axis=0)

    std[std < 1e-8] = 1.0

    return mean.astype(np.float32), std.astype(np.float32)


def normalize_sequences(sequences, mean, std):

    normalized = []

    for sequence in sequences:

        sequence = (sequence - mean) / std

        normalized.append(
            sequence.astype(np.float32)
        )

    return normalized


# ---------------------------------------------------------
# 5. DATASET AND PADDING
# ---------------------------------------------------------

class EyeSequenceDataset(Dataset):

    def __init__(self, sequences, labels):

        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        sequence = torch.tensor(
            self.sequences[index],
            dtype=torch.float32
        )

        label = torch.tensor(
            self.labels[index],
            dtype=torch.long
        )

        return sequence, label


def collate_sequences(batch):

    sequences, labels = zip(*batch)

    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long
    )

    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0
    )

    labels = torch.stack(labels)

    return padded, lengths, labels


# ---------------------------------------------------------
# 6. BI-GRU ENCODER
# ---------------------------------------------------------

class BiGRUEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if NUM_LAYERS > 1 else 0.0
        )

        self.output_size = HIDDEN_SIZE * 2

    def forward(self, sequences, lengths):

        packed = pack_padded_sequence(
            sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, hidden = self.gru(packed)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined = torch.cat(
            [forward_hidden, backward_hidden],
            dim=1
        )

        return combined


# ---------------------------------------------------------
# 7. CONTRASTIVE MODEL
# ---------------------------------------------------------

class ContrastiveModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = BiGRUEncoder()

        self.projector = nn.Sequential(
            nn.Linear(self.encoder.output_size, 128),
            nn.ReLU(),
            nn.Linear(128, PROJECTION_SIZE)
        )

    def forward(self, sequences, lengths):

        features = self.encoder(sequences, lengths)

        projection = self.projector(features)

        projection = nn.functional.normalize(
            projection,
            dim=1
        )

        return projection


# ---------------------------------------------------------
# 8. SUPERVISED CLASSIFIER
# ---------------------------------------------------------

class EyeSenseClassifier(nn.Module):

    def __init__(self):

        super().__init__()

        self.encoder = BiGRUEncoder()

        self.dropout = nn.Dropout(0.3)

        self.classifier = nn.Linear(
            self.encoder.output_size,
            2
        )

    def forward(self, sequences, lengths):

        features = self.encoder(sequences, lengths)

        features = self.dropout(features)

        logits = self.classifier(features)

        return logits


# ---------------------------------------------------------
# 9. AUGMENTATION FOR CONTRASTIVE LEARNING
# ---------------------------------------------------------

def augment_sequences(sequences, lengths):

    augmented = sequences.clone()

    batch_size, max_length, feature_count = augmented.shape

    for i in range(batch_size):

        actual_length = int(lengths[i].item())

        valid_frames = augmented[i, :actual_length]

        # Small Gaussian noise
        noise = torch.randn_like(valid_frames) * 0.05

        valid_frames = valid_frames + noise

        # Mild feature masking
        feature_mask = (
            torch.rand_like(valid_frames) < 0.05
        )

        valid_frames = valid_frames.masked_fill(
            feature_mask,
            0.0
        )

        augmented[i, :actual_length] = valid_frames

    return augmented


# ---------------------------------------------------------
# 10. CONTRASTIVE LOSS (NT-XENT)
# ---------------------------------------------------------

def contrastive_loss(z1, z2):

    batch_size = z1.size(0)

    representations = torch.cat(
        [z1, z2],
        dim=0
    )

    similarity = torch.matmul(
        representations,
        representations.T
    ) / TEMPERATURE

    # Remove self-similarity
    mask = torch.eye(
        batch_size * 2,
        device=DEVICE,
        dtype=torch.bool
    )

    similarity = similarity.masked_fill(
        mask,
        -1e9
    )

    # Positive pair indices
    targets = torch.cat([
        torch.arange(batch_size, 2 * batch_size),
        torch.arange(0, batch_size)
    ]).to(DEVICE)

    loss = nn.functional.cross_entropy(
        similarity,
        targets
    )

    return loss


# ---------------------------------------------------------
# 11. CONTRASTIVE PRETRAINING
# ---------------------------------------------------------

def train_contrastive(model, train_loader):

    print("\n========== PHASE 1: CONTRASTIVE PRETRAINING ==========")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    for epoch in range(CONTRASTIVE_EPOCHS):

        model.train()

        total_loss = 0.0
        batches = 0

        for sequences, lengths, _ in train_loader:

            sequences = sequences.to(DEVICE)

            view1 = augment_sequences(sequences, lengths)
            view2 = augment_sequences(sequences, lengths)

            view1 = view1.to(DEVICE)
            view2 = view2.to(DEVICE)

            optimizer.zero_grad()

            z1 = model(view1, lengths)
            z2 = model(view2, lengths)

            loss = contrastive_loss(z1, z2)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()
            batches += 1

        average_loss = total_loss / max(batches, 1)

        print(
            f"Contrastive Epoch [{epoch + 1}/{CONTRASTIVE_EPOCHS}] "
            f"Loss: {average_loss:.4f}"
        )

    print("Contrastive pretraining completed.")


# ---------------------------------------------------------
# 12. EVALUATION HELPER
# ---------------------------------------------------------

def get_predictions(model, loader):

    model.eval()

    all_labels = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():

        for sequences, lengths, labels in loader:

            sequences = sequences.to(DEVICE)

            logits = model(sequences, lengths)

            probabilities = torch.softmax(logits, dim=1)

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_labels.extend(labels.numpy().tolist())
            all_predictions.extend(predictions.cpu().numpy().tolist())
            all_probabilities.extend(probabilities.cpu().numpy().tolist())

    return (
        np.asarray(all_labels),
        np.asarray(all_predictions),
        np.asarray(all_probabilities)
    )


def calculate_metrics(labels, predictions, probabilities):

    accuracy = accuracy_score(labels, predictions)

    balanced = balanced_accuracy_score(
        labels,
        predictions
    )

    # Class 0 represents Group 1 (Dyslexic).
    auc = roc_auc_score(
        (labels == 0).astype(int),
        probabilities[:, 0]
    )

    return accuracy, balanced, auc


# ---------------------------------------------------------
# 13. SUPERVISED FINE-TUNING
# ---------------------------------------------------------

def train_classifier(model, train_loader, val_loader, train_labels):

    print("\n========== PHASE 2: SUPERVISED FINE-TUNING ==========")

    class_counts = np.bincount(
        train_labels,
        minlength=2
    )

    class_weights = len(train_labels) / (
        2.0 * np.maximum(class_counts, 1)
    )

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=1e-4
    )

    best_score = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(CLASSIFICATION_EPOCHS):

        model.train()

        total_loss = 0.0
        batches = 0

        for sequences, lengths, labels in train_loader:

            sequences = sequences.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()

            logits = model(sequences, lengths)

            loss = criterion(logits, labels)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += loss.item()
            batches += 1

        average_loss = total_loss / max(batches, 1)

        val_labels, val_predictions, val_probabilities = (
            get_predictions(model, val_loader)
        )

        val_accuracy, val_balanced, val_auc = calculate_metrics(
            val_labels,
            val_predictions,
            val_probabilities
        )

        print(
            f"Epoch [{epoch + 1}/{CLASSIFICATION_EPOCHS}] "
            f"Loss: {average_loss:.4f} | "
            f"Val Accuracy: {val_accuracy:.4f} | "
            f"Val Balanced Accuracy: {val_balanced:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        # Early stopping based on validation balanced accuracy
        if val_balanced > best_score:

            best_score = val_balanced

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print("Early stopping activated.")
            break

    if best_state is None:
        raise RuntimeError("No valid model checkpoint was produced.")

    model.load_state_dict(best_state)

    return model


# ---------------------------------------------------------
# 14. FINAL TEST EVALUATION
# ---------------------------------------------------------

def evaluate_test(model, test_loader):

    print("\n========== PHASE 3: FINAL TEST EVALUATION ==========")

    labels, predictions, probabilities = get_predictions(
        model,
        test_loader
    )

    accuracy, balanced, auc = calculate_metrics(
        labels,
        predictions,
        probabilities
    )

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print(f"Test Balanced Accuracy: {balanced:.4f}")
    print(f"Test ROC-AUC: {auc:.4f}")

    print("\nClassification Report:")

    print(
        classification_report(
            labels,
            predictions,
            labels=[0, 1],
            target_names=[
                "Group 1 - Dyslexic",
                "Group 2 - Non-dyslexic"
            ],
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))

    return accuracy, balanced, auc


# ---------------------------------------------------------
# 15. MAIN EXECUTION
# ---------------------------------------------------------

def main():

    set_seed(SEED)

    torch.set_num_threads(4)

    print("\n==========================================")
    print("       EYESENSE CONTRASTIVE LEARNING")
    print("==========================================")

    print("PyTorch version:", torch.__version__)
    print("Device:", DEVICE)

    # Load splits
    print("\nLoading participant sequences...")

    train_sequences, train_labels = load_split(TRAIN_PATH)
    val_sequences, val_labels = load_split(VAL_PATH)
    test_sequences, test_labels = load_split(TEST_PATH)

    print("Training participants:", len(train_sequences))
    print("Validation participants:", len(val_sequences))
    print("Testing participants:", len(test_sequences))

    # Fit normalization using training data only
    mean, std = calculate_normalization(train_sequences)

    train_sequences = normalize_sequences(
        train_sequences, mean, std
    )

    val_sequences = normalize_sequences(
        val_sequences, mean, std
    )

    test_sequences = normalize_sequences(
        test_sequences, mean, std
    )

    print("\nNormalization completed using training data only.")

    # Create datasets
    train_dataset = EyeSequenceDataset(
        train_sequences, train_labels
    )

    val_dataset = EyeSequenceDataset(
        val_sequences, val_labels
    )

    test_dataset = EyeSequenceDataset(
        test_sequences, test_labels
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
        collate_fn=collate_sequences
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sequences
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sequences
    )

    # Phase 1: Contrastive pretraining
    contrastive_model = ContrastiveModel().to(DEVICE)

    train_contrastive(
        contrastive_model,
        train_loader
    )

    # Phase 2: Supervised fine-tuning
    classifier = EyeSenseClassifier().to(DEVICE)

    # Transfer the pretrained encoder weights
    classifier.encoder.load_state_dict(
        contrastive_model.encoder.state_dict()
    )

    classifier = train_classifier(
        classifier,
        train_loader,
        val_loader,
        train_labels
    )

    # Save best model and preprocessing parameters
    torch.save(
        {
            "model_state_dict": classifier.state_dict(),
            "normalization_mean": mean,
            "normalization_std": std,
            "input_size": INPUT_SIZE,
            "hidden_size": HIDDEN_SIZE,
            "num_layers": NUM_LAYERS,
            "group_mapping": {
                "Group 1": "Dyslexic",
                "Group 2": "Non-dyslexic"
            },
            "seed": SEED
        },
        OUTPUT_PATH
    )

    print("\nModel saved to:")
    print(OUTPUT_PATH)

    # Phase 3: Evaluate the test set once
    evaluate_test(
        classifier,
        test_loader
    )

    print("\n==========================================")
    print("       TRAINING PIPELINE COMPLETED")
    print("==========================================")


if __name__ == "__main__":
    main()
