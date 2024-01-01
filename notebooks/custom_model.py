import torch
import numpy as np
from torch import nn
from transformers import AutoModelForSequenceClassification

class AutoCompositeModel(nn.Module):
  def __init__(self, device):
    super(AutoCompositeModel, self).__init__()
    
    self.device = device
    # self.model_content = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=64)
    # self.model_title = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=32)
    
    # encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
    # self.model_content = encoder_layer # nn.TransformerEncoder(encoder_layer, num_layers=6)
    # self.my_new_layers = nn.Sequential(
    #   nn.LayerNorm(128),
    #   nn.Linear(128, 24),
    #   nn.ReLU(),
    #   nn.Linear(24, 128),
    #   nn.ReLU(),
    #   nn.Linear(128, 5)
    # )
    self.lstm = nn.LSTM(128, 128, 2)
    self.test_model = nn.Sequential(
      # nn.LayerNorm(128),
      # # nn.TransformerEncoderLayer(d_model=128, nhead=8),
      # nn.LSTM(128, 32, 2),
      nn.LayerNorm(128),
      nn.Linear(128, 32),
      nn.ReLU(),
      nn.Linear(32, 128),
      nn.ReLU(),
      nn.Linear(128, 5)
    )
  
    # self.my_new_layers = nn.Sequential(
    #   nn.Linear(128, 5),
    # )
  def forward(self, x):

    x = self.lstm(x)[0]
    
    x = self.test_model(x)
    
    return x