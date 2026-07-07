from fastapi import FastAPI
from pydantic import BaseModel
import torch
import torch.nn as nn

# Same brain from train.py
class VerityBrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

app = FastAPI()
model = VerityBrain()
model
