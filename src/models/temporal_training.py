
import os
import random
import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report, confusion_matrix
from temporal_model import EyeSenseGRU


# -------------------------------
# CONFIGURATION
# -------------------------------

SEED = 42
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 0.001
PATIENCE = 5

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cpu")

TRAIN_PATH = "data/processed/splits/train_data.npz"
VAL_PATH = "data/processed/splits/validation_data.npz"

MODEL_DIR = "models/saved"
os.makedirs(MODEL_DIR, exist_ok=True)


# -------------------------------
# LOAD DATA
# -------------------------------

def load_split(path):

    data = np.load(path, allow_pickle=True)

    sequences = [
        np.asarray(seq, dtype=np.float32)
        for seq in data["sequences"]
    ]

    groups = np.asarray(data["groups"], dtype=np.int64)

    return sequences, groups


train_sequences, train_groups = load_split(TRAIN_PATH)
val_sequences, val_groups = load_split(VAL_PATH)

print("\n--- TEMPORAL TRAINING DATA ---")
print("Training participants:", len(train_sequences))
print("Validation participants:", len(val_sequences))


# -------------------------------
# STANDARDISATION
# Fit ONLY on training data
# -------------------------------

all_train_records = np.concatenate(train_sequences, axis=0)

feature_mean = all_train_records.mean(axis=0)
feature_std = all_train_records.std(axis=0)

feature_std[feature_std == 0] = 1.0


def standardise_sequences(sequences):

    return [
        (seq - feature_mean) / feature_std
        for seq in sequences
    ]


train_sequences = standardise_sequences(train_sequences)
val_sequences = standardise_sequences(val_sequences)

np.savez(
    os.path.join(MODEL_DIR, "gru_scaler.npz"),
    mean=feature_mean,
    std=feature_std
)


# -------------------------------
# DATASET
# -------------------------------

class EyeSequenceDataset(Dataset):

    def __init__(self, sequences, groups):

        self.sequences = sequences

        # Convert Group 1 and Group 2 to class indices 0 and 1
        self.labels = groups - 1

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
        [len(seq) for seq in sequences],
        dtype=torch.long
    )

    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0
    )

    labels = torch.stack(labels)

    return padded, lengths, labels


train_dataset = EyeSequenceDataset(
    train_sequences, train_groups
)

val_dataset = EyeSequenceDataset(
    val_sequences, val_groups
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_sequences
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_sequences
)


# -------------------------------
# MODEL
# -------------------------------

model = EyeSenseGRU(
    input_size=4,
    hidden_size=64,
    num_layers=2,
    num_classes=2,
    dropout=0.3
).to(DEVICE)


# Class weights based on training distribution
class_counts = np.bincount(
    train_groups - 1,
    minlength=2
)

class_weights = (
    len(train_groups) / (2.0 * class_counts)
)

criterion = nn.CrossEntropyLoss(
    weight=torch.tensor(
        class_weights,
        dtype=torch.float32
    )
)

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# -------------------------------
# VALIDATION FUNCTION
# -------------------------------

def validate():

    model.eval()

    total_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():

        for sequences, lengths, labels in val_loader:

            outputs = model(sequences, lengths)

            loss = criterion(outputs, labels)

            total_loss += loss.item()

            predicted = torch.argmax(outputs, dim=1)

            predictions.extend(predicted.tolist())
            actuals.extend(labels.tolist())

    average_loss = total_loss / len(val_loader)

    return average_loss, predictions, actuals


# -------------------------------
# TRAINING LOOP
# -------------------------------

best_val_loss = float("inf")
epochs_without_improvement = 0

print("\n--- GRU TRAINING STARTED ---")

for epoch in range(EPOCHS):

    model.train()

    total_train_loss = 0

    for sequences, lengths, labels in train_loader:

        optimizer.zero_grad()

        outputs = model(sequences, lengths)

        loss = criterion(outputs, labels)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=1.0
        )

        optimizer.step()

        total_train_loss += loss.item()

    train_loss = total_train_loss / len(train_loader)

    val_loss, _, _ = validate()

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Validation Loss: {val_loss:.4f}"
    )

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            model.state_dict(),
            os.path.join(MODEL_DIR, "eyesense_gru.pth")
        )

    else:

        epochs_without_improvement += 1

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


# -------------------------------
# FINAL VALIDATION REPORT
# -------------------------------

model.load_state_dict(
    torch.load(
        os.path.join(MODEL_DIR, "eyesense_gru.pth"),
        map_location=DEVICE
    )
)

val_loss, predictions, actuals = validate()

print("\n--- GRU VALIDATION RESULTS ---")
print("Best validation loss:", round(best_val_loss, 4))

print("\nClassification Report:")
print(
    classification_report(
        actuals,
        predictions,
        labels=[0, 1],
        target_names=["Group 1", "Group 2"],
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(actuals, predictions, labels=[0, 1]))

print("\nModel saved: models/saved/eyesense_gru.pth")
print("Scaler saved: models/saved/gru_scaler.npz")
print("\n--- TEMPORAL TRAINING COMPLETED ---")
