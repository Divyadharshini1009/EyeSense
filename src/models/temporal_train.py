
import torch
import torch.nn as nn


class EyeSenseGRU(nn.Module):

    def __init__(
        self,
        input_size=4,
        hidden_size=64,
        num_layers=2,
        num_classes=2,
        dropout=0.3
    ):

        super(EyeSenseGRU, self).__init__()

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        self.dropout = nn.Dropout(dropout)

        self.classifier = nn.Linear(
            hidden_size * 2,
            num_classes
        )

    def forward(self, sequences, lengths):

        packed_sequences = nn.utils.rnn.pack_padded_sequence(
            sequences,
            lengths.cpu(),
            batch_first=True,
            enforce_sorted=False
        )

        _, hidden = self.gru(packed_sequences)

        forward_hidden = hidden[-2]
        backward_hidden = hidden[-1]

        combined_hidden = torch.cat(
            (forward_hidden, backward_hidden),
            dim=1
        )

        combined_hidden = self.dropout(combined_hidden)

        return self.classifier(combined_hidden)


def main():

    print("\n--- EYESENSE TEMPORAL MODEL ---")

    model = EyeSenseGRU()

    print("\nModel architecture:")
    print(model)

    total_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    print("\nTotal trainable parameters:", total_parameters)

    print("\nTraining has not started.")
    print("Dataset validation and temporal-order verification are pending.")


if __name__ == "__main__":
    main()
