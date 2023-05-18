import torch
import numpy as np
import torch.nn.functional as F
from utils.sampling import nucleus_sampling


class Evaluater:
    def __init__(self, model, device):
        self.model = model
        self.device = device

    def calculate_perplexity(self, dataloader, criterion):
        """Calculate perplexity on a data set."""
        self.model.eval()  # Set the model in evaluation mode
        total_loss = 0
        total_count = 0

        with torch.no_grad():
            for data, target in dataloader:
                data, targets = data.to(self.device), target.to(self.device)

                # Initialize and detach the hidden state for each batch
                hidden = self.model.init_hidden(data.size(0))
                hidden = tuple([state.detach() for state in hidden])

                # Forward pass
                outputs, hidden = self.model(data, hidden)
                loss = criterion(outputs, targets.view(-1))
                total_loss += loss.item() * np.prod(targets.size())
                total_count += np.prod(targets.size())

        avg_loss = total_loss / total_count
        perplexity = np.exp(avg_loss)
        return perplexity

    def generate_text(self, seed_text, gen_length, word_to_id, id_to_word, device, temperature=1.0, top_p=1):
        self.model.eval()

        # Convert seed_text to tensor
        input_seq = [word_to_id[word] if word in word_to_id else word_to_id['<UNK>'] for word in seed_text.split()]
        input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)  # Add batch_first dimension
        # input_seq = input_seq.new([word_idx]).unsqueeze(0)

        # Initialize the hidden state
        # hidden = self.model.init_hidden(len(input_seq))
        hidden = self.model.init_hidden(1)

        # Generate text
        generated_text = seed_text
        for _ in range(gen_length):
            with torch.no_grad():
                outputs, hidden = self.model(input_seq, hidden)

                # Apply nucleus sampling to the output logits
                outputs = outputs[-1, :] / temperature
                outputs = nucleus_sampling(outputs, top_p=top_p)

                word_probs = F.softmax(outputs, dim=0)

                # Sample a word from the output probability distribution if the word is not <UNK> otherwise sample again
                word_idx = torch.multinomial(word_probs, 1).item()
                while id_to_word[str(word_idx)] == '<UNK>':
                    word_idx = torch.multinomial(word_probs, 1).item()

                # Append the generated word to the generated text
                generated_word = id_to_word[str(word_idx)]
                generated_text += ' ' + generated_word

                # Update the input sequence with the generated word
                input_seq_text = generated_text.split()[-len(input_seq):]
                input_seq_ids = [word_idx[char] for char in input_seq_text]
                input_seq = torch.tensor(input_seq_ids, dtype=torch.long).unsqueeze(0).to(device)


                # word_idx = torch.multinomial(word_probs, 1).item()

                # # Append the generated word to the generated text
                # generated_word = id_to_word[str(word_idx)]
                # generated_text += ' ' + generated_word

                # # Update the input sequence with the generated word
                # input_seq = torch.tensor([[word_idx]], dtype=torch.long).to(device)

        return generated_text
