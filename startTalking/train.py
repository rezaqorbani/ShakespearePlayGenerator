# train.py

import torch
import torch.nn as nn
import torch.optim as optim

class ModelTrainer:
    def __init__(self, model, dataloader, criterion, optimizer, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self):
        self.model.train()
        hidden = self.model.init_hidden(self.dataloader.batch_size)
        hidden = hidden.to(self.device)
        total_loss = 0
        total_batches = 0
        for inputs, labels in self.dataloader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            output, _ = self.model(inputs, hidden)
            loss = self.criterion(output,labels.view(-1))
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            total_batches += 1
        epoch_loss = total_loss / total_batches

        return epoch_loss

    def evaluate(self, validation_loader):
        self.model.eval()
        total_loss = 0
        total_batches = 0
        # Initialize the hidden state
        hidden = self.model.init_hidden(validation_loader.batch_size)
        hidden = hidden.to(self.device)

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Forward pass
                outputs, hidden = self.model(inputs, hidden)

                # Compute the loss
                loss = self.criterion(outputs, labels.view(-1))
                total_loss += loss.item()
                total_batches += 1

        epoch_loss = total_loss / total_batches
        return epoch_loss
    
    
