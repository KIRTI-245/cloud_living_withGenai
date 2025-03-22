from torch import nn
import torch

class TransformerPricePredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.transformer = nn.Transformer(d_model=64, nhead=8, num_encoder_layers=2)
        self.fc = nn.Linear(64, 1)
    
    def forward(self, src, tgt):
        out = self.transformer(src, tgt)
        return self.fc(out)

# Sample rental price data (time-series)
src = torch.rand((10, 1, 64))  # 10 months historical data
tgt = torch.rand((10, 1, 64))  # target placeholder

model = TransformerPricePredictor()
prediction = model(src, tgt)
print(prediction)
