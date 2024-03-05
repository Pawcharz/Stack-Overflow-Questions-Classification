import torch
import numpy as np
from torch import nn
from transformers import AutoModelForSequenceClassification

# 'content_input_ids', 'title_input_ids', 'tags_onehot', 'unrecognized_tags_count', 'reputation', 'undeleted_answers', 'user_life_days'

class TagsModel(nn.Module):
  def __init__(self, device):
    super(TagsModel, self).__init__()
    
    self.device = device
    
    self.forward_layer = nn.Sequential(
      nn.Linear(100, 86),
      nn.ReLU(),
      # nn.Linear(86, 32),
      # nn.ReLU(),
      nn.Linear(86, 5)
    )
  def forward(self, inputs):
        
    tags_input = inputs['tags_onehot'].to(self.device).float()
   
    x = self.forward_layer(tags_input)
    
    return x
  
# 'unrecognized_tags_count', 'reputation', 'undeleted_answers', 'user_life_days'
class NumericalPartModel(nn.Module):
  def __init__(self, device):
    super(NumericalPartModel, self).__init__()
    
    self.device = device
    
    # self.forward_layer = nn.Sequential(
    #   nn.Linear(4, 128),
    #   nn.ReLU(),
    #   nn.Linear(128, 256),
    #   nn.ReLU(),
    #   nn.Linear(256, 128),
    #   nn.ReLU(),
    #   nn.Linear(128, 5),
    # )
    
    self.forward_layer = nn.Sequential(
      nn.Linear(4, 16),
      nn.ReLU(),
      nn.Linear(16, 16),
      nn.ReLU(),
      nn.Linear(16, 5),
    )
  def forward(self, inputs):
        
    unrec_tags_input = inputs['unrecognized_tags_count'].to(self.device).float()
    reputation_input = inputs['reputation'].to(self.device).float()
    answers_input = inputs['undeleted_answers'].to(self.device).float()
    life_input = inputs['user_life_days'].to(self.device).float()
   
    all = torch.stack((unrec_tags_input, reputation_input, answers_input, life_input), 1)
    
    x = self.forward_layer(all)
    
    return x

# 'content_input_ids', 'title_input_ids',
class TextualPartModel(nn.Module):
  def __init__(self, device):
    super(TextualPartModel, self).__init__()
    
    self.device = device
    
    # input - 32
    self.titleEncoder = nn.TransformerEncoderLayer(d_model=32, nhead=2, norm_first=True)
    
    # input - 128
    self.contentEncoder = nn.TransformerEncoderLayer(d_model=128, nhead=4, norm_first=True)
    
    self.forward_layer = nn.Sequential(
      nn.Linear(160, 86),
      nn.ReLU(),
      nn.Linear(86, 5),
    )
  def forward(self, inputs):
        
    title_input = inputs['title_input_ids'].to(self.device).float()
    content_input = inputs['content_input_ids'].to(self.device).float()
    
    title_input = self.titleEncoder(title_input)
    content_input = self.contentEncoder(content_input)
   
    all = torch.cat((title_input, content_input), 1)
    
    x = self.forward_layer(all)
    
    return x

 
class AutoCompositeModel(nn.Module):
  def __init__(self, device):
    super(AutoCompositeModel, self).__init__()
    
    self.device = device
    
    self.forward_layer = nn.Sequential(
      nn.Linear(5, 32),
      nn.ReLU(),
      nn.Linear(32, 12),
      nn.ReLU(),
      nn.Linear(12, 5)
    )
    
    self.tags_layer = nn.Sequential(
      nn.Linear(100, 48),
      nn.Linear(48, 1),
    )
    
  def forward(self, inputs):
    
    # 'unrecognized_tags_count', 'reputation', 'tags_onehot', 'undeleted_answers', 'user_life_days'
    
    tags_onehot = inputs['tags_onehot'].reshape(-1, 1).to(self.device).float()
    print(tags_onehot.size(), tags_onehot)
    
    tags_output = self.tags_layer(tags_onehot)
    
    urecognized_tags = inputs['unrecognized_tags_count'].reshape(-1, 1).to(self.device).float()
    reputation = inputs['reputation'].reshape(-1, 1).to(self.device).float()
    undeleted = inputs['undeleted_answers'].reshape(-1, 1).to(self.device).float()
    life_days = inputs['user_life_days'].reshape(-1, 1).to(self.device).float()
    
    all = torch.cat((reputation, undeleted, life_days, urecognized_tags, tags_output), 1)
  
    x = self.forward_layer(all)
    
    return x