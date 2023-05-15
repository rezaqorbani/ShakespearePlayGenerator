import torch
import numpy as np
import torch.nn.functional as F


class Evaluater:
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    
    def calculate_perplexity(self, dataloader, criterion):
        """Calculate perplexity on a data set."""
        self.model.eval()  # Set the model in evaluation mode
        total_loss = 0
        total_count = 0
        hidden = self.model.init_hidden(dataloader.batch_size)

        with torch.no_grad():
            for data, target in dataloader:
                data, targets = data.to(self.device), target.to(self.device)

                # Detach the hidden state from the computation graph to prevent backpropagation through time
                hidden = tuple([state.detach() for state in hidden])

                # Forward pass
                outputs, hidden = self.model(data, hidden)
                loss = criterion(outputs, targets.view(-1))
                total_loss += loss.item() * np.prod(targets.size())
                total_count += np.prod(targets.size())

        avg_loss = total_loss / total_count
        perplexity = np.exp(avg_loss)
        return perplexity
    
    def generate_text(self, seed_text, gen_length, char_to_id, id_to_char, device, temperature=1.0):
        self.model.eval()

        # Convert seed_text to tensor
        input_seq = [char_to_id[char] for char in seed_text]
        input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)  # Add batch_first dimension

        # Initialize the hidden state
        hidden = self.model.init_hidden(1)

        # Generate text
        generated_text = seed_text
        for _ in range(gen_length):
            with torch.no_grad():
                outputs, hidden = self.model(input_seq, hidden)
                char_probs = F.softmax(outputs[-1, :] / temperature, dim=0)

                # Sample a character from the output probabilities
                char_idx = torch.multinomial(char_probs, 1).item()

                # Append the generated character to the generated text
                generated_char = id_to_char[str(char_idx)]
                generated_text += generated_char

                # Update the input sequence with the generated character
                input_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)

        return generated_text
    
    import torch.nn.functional as F

def generate_text(model, seed_text, gen_length, char_to_id, id_to_char, device, temperature=1.0):
        model.eval()

        # Convert seed_text to tensor
        input_seq = [char_to_id[char] for char in seed_text]
        input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)  # Add batch_first dimension

        # Initialize the hidden state
        hidden = model.init_hidden(1)

        # Generate text
        generated_text = seed_text
        for _ in range(gen_length):
            with torch.no_grad():
                outputs, hidden = model(input_seq, hidden)
                char_probs = F.softmax(outputs[-1, :] / temperature, dim=0)

                # Sample a character from the output probabilities
                char_idx = torch.multinomial(char_probs, 1).item()

                # Append the generated character to the generated text
                generated_char = id_to_char[str(char_idx)]
                generated_text += generated_char

                # Update the input sequence with the generated character
                input_seq = torch.tensor([[char_idx]], dtype=torch.long).to(device)

        return generated_text