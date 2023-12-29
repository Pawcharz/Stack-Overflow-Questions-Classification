import torch
import numpy as np
from datetime import datetime

# PyTorch TensorBoard support
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

# Initializing in a separate cell so we can easily add more epochs to the same run
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
sum_writer = SummaryWriter('runs/chest_trainer_{}'.format(timestamp))

class TrainerConfiguration():
  def __init__(self, training_loader, validation_loader, optimizer, loss_fn, accuracy_metric, device):
    
    self.training_loader = training_loader
    self.validation_loader = validation_loader
    
    self.optimizer = optimizer
    
    self.loss_fn = loss_fn
    self.accuracy_metric = accuracy_metric
    
    self.device = device

class Trainer():
  def __init__(self, model, trainer_configuration: TrainerConfiguration, input_column, output_column, epoch_index=0):
    
    self.config = trainer_configuration
    self.model = model.to(self.config.device)
    
    self.input_column = input_column
    self.output_column = output_column
    self.epoch_index = epoch_index
    
    
  def train_one_epoch(self, logging_frequency, evaluate_when_logging: bool):
    running_loss = 0.
    running_accuracy_all = 0.
    running_accuracy = 0.
    last_loss = 0.

    # Here, we use enumerate(training_loader) instead of
    # iter(training_loader) so that we can track the batch
    # index and do some intra-epoch reporting
    for i, data in enumerate(self.config.training_loader):
    
    # torch.tensor(tokenized_test_dataset['input_ids'][:10]).float().to(device)
      # Every data instance is an input + label pair
      
      # print(data[self.input_column])
      inputs = torch.stack(data[self.input_column], 1).float().to(self.config.device)
      # print(inputs)
      labels = data[self.output_column].to(self.config.device) # torch.stack(data[self.output_column]).float().to(self.config.device)
    
      # Zero your gradients for every batch!
      self.config.optimizer.zero_grad()

      # Make predictions for this batch
      outputs = self.model(inputs)

      # Compute the loss and its gradients
      loss = self.config.loss_fn(outputs, labels)
      loss.backward()
      
      training_accuracy = self.config.accuracy_metric(outputs, labels)

      # Adjust learning weights
      self.config.optimizer.step()

      # Gather data and report
      running_loss += loss.item()
      
      # print(running_loss)
      running_accuracy += training_accuracy
      running_accuracy_all += training_accuracy
      # print('batch {}', i)
      
      if (i+1) % logging_frequency == 0:
        last_loss = running_loss / logging_frequency # loss per batch
        last_accuracy = running_accuracy / logging_frequency # accuracy per batch
        
        if evaluate_when_logging == True:
          avg_vloss, avg_vacc = self.evaluate_model()
          print(' batch {} training_loss: {} validation_loss: {} training_accuracy: {} validation_accuracy {}'.format(i + 1, last_loss, avg_vloss, last_accuracy, avg_vacc))
        else:
          print(' batch {} training_loss: {} training_accuracy: {}'.format(i + 1, last_loss, last_accuracy))
          
        tb_x = self.epoch_index * len(self.config.training_loader) + i + 1
        sum_writer.add_scalar('Loss/train', last_loss, tb_x)
        running_loss = 0.
        running_accuracy = 0.
    
    accuracy_all = running_accuracy_all / len(self.config.training_loader)
    
    self.epoch_index += 1
    return last_loss, accuracy_all

  def evaluate_model(self):
    
    running_vloss = 0.0
    running_vacc = 0.0
    
    # Set the model to evaluation mode, disabling dropout and using population
    # statistics for batch normalization.
    self.model.eval()
    
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
      for i, vdata in enumerate(self.config.validation_loader):
        vinputs = torch.stack(vdata[self.input_column], 1).float().to(self.config.device)
        vlabels = vdata[self.output_column].to(self.config.device)
        voutputs = self.model(vinputs)
        
        vloss = self.config.loss_fn(voutputs, vlabels)
        running_vloss += vloss
        
        vacc = self.config.accuracy_metric(voutputs, vlabels)
        running_vacc += vacc

    avg_vloss = running_vloss / (i + 1)
    avg_vacc = running_vacc / (i + 1)
    
    return [avg_vloss, avg_vacc]



def train_many_epochs(epochs, model, training_loader, validation_loader, optimizer, loss_fn, accuracy_metric, cuda_device, epoch_index, logging_frequency):
  best_vloss = 1_000_000.
  timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

  for epoch_number in range(epochs):
    print('EPOCH {}:'.format(epoch_number + 1))

    # Make sure gradient tracking is on, and do a pass over the data
    model.train(True)
    avg_loss, avg_acc = train_one_epoch(
      model=model,
      training_loader=training_loader,
      validation_loader=validation_loader,
      optimizer=optimizer,
      loss_fn=loss_fn,
      accuracy_metric=accuracy_metric,
      cuda_device=cuda_device,
      epoch_index=epoch_index,
      logging_frequency=logging_frequency
    )
   

    avg_vloss, avg_vacc = evaluate_model(
      model=model,
      validation_loader=validation_loader,
      loss_fn=loss_fn,
      accuracy_metric=accuracy_metric,
      cuda_device=cuda_device
    )
    print('LOSS train {} valid {} ACCURACY train {} validation {}'.format(avg_loss, avg_vloss, avg_acc, avg_vacc))

    # Log the running loss averaged per batch
    # for both training and validation
    sum_writer.add_scalars(
      'Training vs. Validation Loss',
      { 'Training' : avg_loss, 'Validation' : avg_vloss },
      epoch_number + 1
    )
    sum_writer.flush()

    # Track best performance, and save the model's state
    if avg_vloss < best_vloss:
      best_vloss = avg_vloss
      model_path = 'model_{}_{}'.format(timestamp, epoch_number)
      torch.save(model.state_dict(), model_path)

    epoch_number += 1
    

def get_model_params(model):
  pp=0
  for p in list(model.parameters()):
    nn=1
    for s in list(p.size()):
      nn = nn*s
    pp += nn
  return pp
  