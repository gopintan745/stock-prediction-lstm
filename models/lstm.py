import torch
from torch import nn

class StockLSTM(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, output_size: int = 1, dropout: float = 0.2):
        super(StockLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, input_size)
        out, (h_n, c_n) = self.lstm(x)
        last_hidden = out[:, -1, :]          # (batch, hidden_size)
        last_hidden = self.dropout(last_hidden)
        pred = self.fc(last_hidden)          # (batch, 1)
        return pred.squeeze(-1)              # (batch,)