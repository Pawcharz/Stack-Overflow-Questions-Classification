import torch
from torch import nn
from transformers import AutoModelForSequenceClassification

class AutoCompositeModel(nn.Module):
  def __init__(self, device):
    super(AutoCompositeModel, self).__init__()
    
    self.device = device
    # self.model_content = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=64)
    # self.model_title = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=32)
    
    encoder_layer = nn.TransformerEncoderLayer(d_model=128, nhead=8)
    self.model_content = nn.TransformerEncoder(encoder_layer, num_layers=6)
    self.my_new_layers = nn.Sequential(
      nn.LayerNorm(128),
      nn.Linear(128, 24),
      nn.ReLU(),
      nn.Linear(24, 128),
      nn.ReLU(),
      nn.Linear(128, 5)
    )
  
    # self.my_new_layers = nn.Sequential(
    #   nn.Linear(128, 5),
    # )
  def forward(self, x):
    # x = self.model_content(x).to(self.device)
    # x = self.model_content(x).logits.to(self.device)
    x = self.my_new_layers(x)
    
    return x