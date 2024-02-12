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
    
    # self.lstm = nn.LSTM(128, 128, 3)
    
    self.contentEncoder = nn.TransformerEncoderLayer(d_model=128, nhead=8, norm_first=True)
    self.titleEncoder = nn.TransformerEncoderLayer(d_model=32, nhead=8, norm_first=True)
    
    self.test_model = nn.Sequential(
      nn.Linear(163, 256),
      nn.ReLU(),
      nn.Linear(256, 512),
      nn.ReLU(),
      nn.Linear(512, 512),
      nn.ReLU(),
      nn.Linear(512, 128),
      nn.ReLU(),
      nn.Linear(128, 256),
      nn.ReLU(),
      nn.Linear(256, 256),
      nn.ReLU(),
      nn.Linear(256, 128),
      nn.ReLU(),
      nn.Linear(128, 5)
    )
  def forward(self, inputs):
        
    text_content = inputs['content_input_ids'].to(self.device).float()
    title = inputs['title_input_ids'].to(self.device).float()
    
    reputation = inputs['reputation'].reshape(-1, 1).to(self.device).float()
    undeleted = inputs['undeleted_answers'].reshape(-1, 1).to(self.device).float()
    life_days = inputs['user_life_days'].reshape(-1, 1).to(self.device).float()
    
    x_content = self.contentEncoder(text_content).float()
    x_title = self.titleEncoder(title).float()
    rest = torch.cat((reputation, undeleted, life_days), 1)
    
    x_concated = torch.cat((x_content, x_title, rest), 1)
    x = self.test_model(x_concated)
    
    return x