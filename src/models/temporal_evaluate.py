
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)

from temporal_model import EyeSenseGRU


# Configuration
DEVICE = torch.device("cpu")
BATCH_SIZE = 32

TEST_PATH = "data/processed/splits/test_data.npz"
SCALER_PATH = "models/saved/gru_scaler.npz"
MODEL_PATH = "models/saved/eyesense_gru.pth"


# Load test data
data = np.load(TEST_PATH, allow_pickle=True)

sequences = [
    np.asarray(seq, dtype=np.float32)
    for seq in data["sequences"]
]

groups = np.asarray(data["groups"], dtype=np.int64)

# Load training-set scaler
scaler = np.load(SCALER_PATH)

mean = scaler["mean"]
std = scaler["std"]

sequences = [
    (seq - mean) / std
    for seq in sequences
]

print("\n--- GRU TEST DATA ---")
print("Test participants:", len(sequences))


# Dataset
class TestDataset(Dataset):

    def __init__(self, sequences, groups):
        self.sequences = sequences
        self.labels = groups - 1

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return (
            torch.tensor(
                self.sequences[index],
                dtype=torch.float32
            ),
            torch.tensor(
                self.labels[index],
                dtype=torch.long
            )
        )


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

    return padded, lengths, torch.stack(labels)


dataset = TestDataset(sequences, groups)

loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_sequences
)


# Load trained GRU
model = EyeSenseGRU(
    input_size=4,
    hidden_size=64,
    num_layers=2,
    num_classes=2,
    dropout=0.3
).to(DEVICE)

model.load_state_dict(
    torch.load(MODEL_PATH, map_location=DEVICE)
)

model.eval()


# Evaluate
actuals = []
predictions = []
probabilities = []

with torch.no_grad():

    for batch, lengths, labels in loader:

        outputs = model(batch, lengths)

        probs = torch.softmax(outputs, dim=1)

        predicted = torch.argmax(outputs, dim=1)

        actuals.extend(labels.tolist())
        predictions.extend(predicted.tolist())
        probabilities.extend(probs[:, 1].tolist())


# Metrics
accuracy = accuracy_score(actuals, predictions)

balanced_accuracy = balanced_accuracy_score(
    actuals, predictions
)

auc = roc_auc_score(actuals, probabilities)

print("\n--- GRU TEST RESULTS ---")

print("Accuracy:", round(accuracy, 4))
print("Balanced Accuracy:", round(balanced_accuracy, 4))
print("ROC-AUC:", round(auc, 4))

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

print(
    confusion_matrix(
        actuals,
        predictions,
        labels=[0, 1]
    )
)

print("\n--- GRU TEST EVALUATION COMPLETED ---")
