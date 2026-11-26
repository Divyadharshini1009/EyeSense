
import torch
import torch.nn as nn


class EyeSenseGRU(nn.Module):
    """
    GRU-based temporal model for eye-tracking sequences.
    """

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

        output = self.classifier(combined_hidden)

        return output


if __name__ == "__main__":

    print("\n--- EYESENSE GRU MODEL TEST ---")

    model = EyeSenseGRU()

    # Example batch: 3 participants, maximum 10 fixations each
    sample_sequences = torch.randn(3, 10, 4)

    # Actual sequence lengths before padding
    sequence_lengths = torch.tensor([10, 8, 6])

    output = model(sample_sequences, sequence_lengths)

    print("Input shape:", sample_sequences.shape)
    print("Output shape:", output.shape)
    print("Model parameters:", sum(
        parameter.numel()
        for parameter in model.parameters()
    ))

    print("\nGRU architecture initialized successfully!")
