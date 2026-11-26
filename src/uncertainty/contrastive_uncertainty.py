
"""
EyeSense - Monte Carlo Dropout Uncertainty Quantification
For the trained Contrastive Bi-GRU model.

Group 1 = Dyslexic
Group 2 = Non-dyslexic
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.nn.utils.rnn import (
    pad_sequence,
    pack_padded_sequence
)
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import accuracy_score


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

SEED = 42
BATCH_SIZE = 64
MC_ITERATIONS = 30

INPUT_SIZE = 4
HIDDEN_SIZE = 64
NUM_LAYERS = 2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = ROOT / "models" / "saved" / "contrastive_bigru_best.pt"

VALIDATION_PATH = (
    ROOT / "data" / "processed" / "splits" /
    "validation_data.npz"
)

OUTPUT_FOLDER = (
    ROOT / "data" / "processed" / "uncertainty"
)

OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# MODEL ARCHITECTURE
# Must match the trained Bi-GRU architecture.
# --------------------------------------------------

class BiGRUEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.gru = nn.GRU(
            input_size=INPUT_SIZE,
            hidden_size=HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
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

        return torch.cat(
            [forward_hidden, backward_hidden],
            dim=1
        )


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

        return self.classifier(features)


# --------------------------------------------------
# DATASET
# --------------------------------------------------

class EyeSequenceDataset(Dataset):

    def __init__(self, sequences, groups, subject_ids):

        self.sequences = sequences
        self.groups = groups
        self.subject_ids = subject_ids

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):

        return (
            torch.tensor(
                self.sequences[index],
                dtype=torch.float32
            ),
            int(self.groups[index]),
            str(self.subject_ids[index])
        )


def collate_sequences(batch):

    sequences, groups, subject_ids = zip(*batch)

    lengths = torch.tensor(
        [len(sequence) for sequence in sequences],
        dtype=torch.long
    )

    padded = pad_sequence(
        sequences,
        batch_first=True,
        padding_value=0.0
    )

    groups = torch.tensor(groups, dtype=torch.long)

    return padded, lengths, groups, subject_ids


# --------------------------------------------------
# LOAD AND PREPROCESS VALIDATION DATA
# --------------------------------------------------

def load_validation_data(checkpoint):

    data = np.load(
        VALIDATION_PATH,
        allow_pickle=True
    )

    sequences = list(data["sequences"])

    groups = np.asarray(
        data["groups"],
        dtype=np.int64
    )

    subject_ids = np.asarray(
        data["subject_ids"]
    )

    mean = checkpoint["normalization_mean"]
    std = checkpoint["normalization_std"]

    processed = []

    for sequence in sequences:

        sequence = np.asarray(
            sequence,
            dtype=np.float32
        ).copy()

        # Same duration transformation used during training
        sequence[:, 3] = np.log1p(sequence[:, 3])

        # Same training-set normalization
        sequence = (sequence - mean) / std

        processed.append(
            sequence.astype(np.float32)
        )

    return processed, groups, subject_ids


# --------------------------------------------------
# MONTE CARLO DROPOUT
# --------------------------------------------------

def monte_carlo_predictions(model, loader):

    all_results = []

    for sequences, lengths, groups, subject_ids in loader:

        sequences = sequences.to(DEVICE)

        stochastic_probabilities = []

        # Dropout remains active for repeated predictions.
        # Gradients are disabled.
        model.train()

        with torch.no_grad():

            for _ in range(MC_ITERATIONS):

                logits = model(
                    sequences,
                    lengths
                )

                probabilities = torch.softmax(
                    logits,
                    dim=1
                )

                stochastic_probabilities.append(
                    probabilities.cpu().numpy()
                )

        # Shape: MC_ITERATIONS x batch_size x 2
        mc_probabilities = np.stack(
            stochastic_probabilities,
            axis=0
        )

        all_results.append(
            (
                mc_probabilities,
                groups.numpy(),
                list(subject_ids)
            )
        )

    return all_results


# --------------------------------------------------
# UNCERTAINTY CALCULATIONS
# --------------------------------------------------

def calculate_entropy(probabilities):

    probabilities = np.clip(
        probabilities,
        1e-10,
        1.0
    )

    return -np.sum(
        probabilities * np.log2(probabilities),
        axis=-1
    )


def generate_report(all_results):

    reports = []

    for mc_probs, groups, subject_ids in all_results:

        # Average probability across stochastic passes
        mean_probabilities = mc_probs.mean(axis=0)

        # Predictive entropy
        predictive_entropy = calculate_entropy(
            mean_probabilities
        )

        # Average entropy of individual predictions
        individual_entropy = calculate_entropy(
            mc_probs
        )

        expected_entropy = individual_entropy.mean(axis=0)

        # Mutual information:
        # Measures uncertainty caused by model variation.
        mutual_information = np.maximum(
            predictive_entropy - expected_entropy,
            0.0
        )

        predictions = np.argmax(
            mean_probabilities,
            axis=1
        )

        confidence = np.max(
            mean_probabilities,
            axis=1
        )

        # Heuristic uncertainty categories.
        # These thresholds are not clinical cut-offs.
        uncertainty_levels = []

        for entropy_value in predictive_entropy:

            if entropy_value < 0.30:
                uncertainty_levels.append("Low")

            elif entropy_value < 0.60:
                uncertainty_levels.append("Moderate")

            else:
                uncertainty_levels.append("High")

        results = pd.DataFrame({

            "SubjectID": subject_ids,

            "Actual_Group": groups,

            "Predicted_Group": predictions + 1,

            "Dyslexia_Probability": mean_probabilities[:, 0],

            "Non_Dyslexia_Probability": mean_probabilities[:, 1],

            "Prediction_Confidence": confidence,

            "Predictive_Entropy": predictive_entropy,

            "Expected_Entropy": expected_entropy,

            "Mutual_Information": mutual_information,

            "Uncertainty_Level": uncertainty_levels

        })

        reports.append(results)

    return pd.concat(
        reports,
        ignore_index=True
    )


# --------------------------------------------------
# MAIN
# --------------------------------------------------

def main():

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    torch.set_num_threads(4)

    print("\n========================================")
    print(" EYESENSE UNCERTAINTY QUANTIFICATION")
    print("========================================")

    print("Device:", DEVICE)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    if not VALIDATION_PATH.exists():
        raise FileNotFoundError(
            f"Validation data not found: {VALIDATION_PATH}"
        )

    print("\nLoading trained Contrastive Bi-GRU model...")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model = EyeSenseClassifier().to(DEVICE)

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print("Model loaded successfully.")

    print("\nLoading validation sequences...")

    sequences, groups, subject_ids = load_validation_data(
        checkpoint
    )

    dataset = EyeSequenceDataset(
        sequences,
        groups,
        subject_ids
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_sequences
    )

    print("Validation participants:", len(dataset))

    print("\nRunning Monte Carlo Dropout...")
    print("Repeated predictions per participant:", MC_ITERATIONS)

    all_results = monte_carlo_predictions(
        model,
        loader
    )

    print("\nCalculating uncertainty measures...")

    report = generate_report(all_results)

    # Save participant-level results
    output_path = (
        OUTPUT_FOLDER /
        "contrastive_validation_uncertainty.csv"
    )

    report.to_csv(
        output_path,
        index=False
    )

    accuracy = accuracy_score(
        report["Actual_Group"],
        report["Predicted_Group"]
    )

    print("\n========== RESULTS ==========")

    print("Participants analysed:", len(report))

    print(
        "Prediction accuracy:",
        round(accuracy, 4)
    )

    print("\nUncertainty distribution:")

    print(
        report["Uncertainty_Level"].value_counts()
    )

    print(
        "\nAverage prediction confidence:",
        round(report["Prediction_Confidence"].mean(), 4)
    )

    print(
        "Average predictive entropy:",
        round(report["Predictive_Entropy"].mean(), 4)
    )

    print(
        "Average mutual information:",
        round(report["Mutual_Information"].mean(), 4)
    )

    print("\nFirst five predictions:")

    print(
        report.head().to_string(index=False)
    )

    print("\nReport saved to:")
    print(output_path)

    print("\nUNCERTAINTY QUANTIFICATION COMPLETED")


if __name__ == "__main__":
    main()
