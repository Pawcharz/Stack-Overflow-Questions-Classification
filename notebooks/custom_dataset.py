import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

def encodeTags(dataframe):
  tags_columns = ['Tag1', 'Tag2', 'Tag3', 'Tag4', 'Tag5']
  encoded = pd.get_dummies(dataframe['Tag1'], columns=['Tag1'], prefix='tag_')

  for i in range(1, 4):
    
    col_name = tags_columns[i]
    dummies = pd.get_dummies(dataframe[col_name], columns=[col_name], prefix='tag_')
    encoded |= dummies
    # print(i, dummies['tag__86.0'], encoded['tag__86.0'])
    
  return encoded

class SingleGithubExample:
  def __init__(self, tags_onehot, unrecognized_tags_count, reputation, undeleted_answers, user_life_days, title, text_content):
    self.tags_onehot = tags_onehot
    self.unrecognized_tags_count = unrecognized_tags_count
    self.reputation = reputation
    self.undeleted_answers = undeleted_answers
    self.user_life_days = user_life_days
    self.title = title
    self.text_content = text_content
    

class GithubDataset(Dataset):
  def __init__(self, df):
    self.tags_onehot = []
    self.unrecognized_tags_count = []
    self.reputation = []
    self.undeleted_answers = []
    self.user_life_days = []
    
    self.title = []
    self.text_content = []

    # Encoding tags
    tags_onehot_pd = encodeTags(df)
    tags_onehot = torch.from_numpy(tags_onehot_pd.to_numpy())
    self.tags_onehot = tags_onehot
    self.unrecognized_tags_count = torch.from_numpy(df['UnrecognizedTags'].to_numpy())
    
    # Saving the rest of the numerical data
    self.reputation = torch.from_numpy(df['ReputationAtPostCreation'].to_numpy())
    self.undeleted_answers = torch.from_numpy(df['OwnerUndeletedAnswerCountAtPostTime'].to_numpy())
    self.user_life_days = torch.from_numpy(df['DaysTillPosting'].to_numpy())
    
    # Saving Text
    self.title = df['Title']
    self.text_content = df['BodyMarkdown']
    
  def __len__(self):
    return len(self.tags_onehot)

  def __getitem__(self, i):
    return SingleGithubExample(
      self.tags_onehot[i],
      self.unrecognized_tags_count[i],
      self.reputation[i],
      self.undeleted_answers[i],
      self.user_life_days[i],
      self.title[i],
      self.text_content[i]
    )