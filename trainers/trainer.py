import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

class ModelTrainer:
    def __init__(self, model, dataloader, criterion, optimizer, device):
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device

    def train(self):
        self.model.train()
        total_loss = 0
        total_batches = 0
        # for inputs, labels in self.dataloader:
        for inputs, labels in tqdm(self.dataloader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # Initialize and detach the hidden state for each batch
            hidden = self.model.init_hidden(inputs.size(0))
            hidden = tuple([state.detach() for state in hidden])

            # Forward pass
            outputs, hidden = self.model(inputs, hidden)
            
            # Compute the loss
            loss = self.criterion(outputs, labels.view(-1))
            total_loss += loss.item()
            total_batches += 1
            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
        epoch_loss = total_loss / total_batches

        return epoch_loss

    def evaluate(self, validation_loader):
        self.model.eval()
        total_loss = 0
        total_batches = 0

        with torch.no_grad():
            for inputs, labels in validation_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                # Initialize and detach the hidden state for each batch
                hidden = self.model.init_hidden(inputs.size(0))
                hidden = tuple([state.detach() for state in hidden])

                # Forward pass
                outputs, hidden = self.model(inputs, hidden)

                # Compute the loss
                loss = self.criterion(outputs, labels.view(-1))
                total_loss += loss.item()
                total_batches += 1

        epoch_loss = total_loss / total_batches
        return epoch_loss
    
    
