from torch import nn
from transformers import AutoModelForSequenceClassification

class AutoCompositeModel(nn.Module):
  def __init__(self, device):
    super(AutoCompositeModel, self).__init__()
    
    self.device = device
    self.model_content = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=64)
    # self.model_title = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=32)
    
    self.my_new_layers = nn.Sequential(
      # nn.LayerNorm(128),
      nn.Linear(64, 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, 5)
    )
  
  def forward(self, x):
    x = self.model_content(x).logits.to(self.device)
    # x = self.model_content(x).logits.to(self.device)
    x = self.my_new_layers(x)
    
    return x