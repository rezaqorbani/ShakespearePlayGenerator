import torch
import numpy as np
import torch.nn.functional as F
from utils.sampling import nucleus_sampling
import re



class Evaluater:
    def __init__(self, model, device, number_states=2):
        self.model = model
        self.device = device
        self.number_states = number_states

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
                if self.number_states==2:
                  hidden = tuple([state.detach() for state in hidden])
                else:
                  hidden = hidden.detach()

                # Forward pass
                outputs, hidden = self.model(data, hidden)
                loss = criterion(outputs, targets.view(-1))
                total_loss += loss.item() * np.prod(targets.size())
                total_count += np.prod(targets.size())

        avg_loss = total_loss / total_count
        perplexity = np.exp(avg_loss)
        return perplexity

    def generate_text(self, seed_text, gen_length, token_to_id, id_to_token, level, device, temperature=1.0, top_p=1, tokenizer=None):
        self.model.eval()
        if level=='char':
          input=seed_text
        elif level == 'word':
          input=seed_text.split()
        elif level == 'bpe':
          input=tokenizer.tokenizer.encode(seed_text).tokens
          
        # Convert seed_text to tensor
        input_seq = [token_to_id[word] if word in token_to_id else token_to_id['<UNK>'] for word in input]
        input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)  # Add batch_first dimension
        # input_seq = input_seq.new([word_idx]).unsqueeze(0)

        # Initialize the hidden state
        # hidden = self.model.init_hidden(len(input_seq))
        hidden = self.model.init_hidden(1)

        # Generate text
        generated_text = seed_text

        for _ in range(gen_length):#

            with torch.no_grad():
                outputs, hidden = self.model(input_seq, hidden)

                # Apply nucleus sampling to the output logits
                outputs = outputs[-1, :] / temperature
                outputs = nucleus_sampling(outputs, top_p=top_p)

                probs = F.softmax(outputs, dim=0)
                # Sample a word from the output probability distribution if the word is not <UNK> otherwise sample again
                idx = torch.multinomial(probs, 1).item()
                # while id_to_[str(idx)] == '<UNK>':
                # while (idx not in id_to_token) or (id_to_token[str(idx)] == '<UNK>'):
                #     idx = torch.multinomial(probs, 1).item()
                #     print(idx)
                
                
                while idx not in id_to_token or id_to_token[idx] == '<UNK>' :
                  idx = torch.multinomial(probs, 1).item()
                  
                # Append the generated word to the generated text
                # generated_ = id_to_[str(idx)]
                generated_ = id_to_token[idx]

                # Update the input sequence with the generated word
                if level=='word':
                  generated_text += ' ' + generated_
                  input_seq_text = generated_text.split()[-len(input_seq):]
                elif level=='char':
                  generated_text += generated_
                  input_seq_text = generated_text[-len(input_seq):]
                elif level=='bpe':
                  generated_text += ' ' + generated_
                  # generated_ids = tokenizer.tokenizer.encode(generated_text).ids
                  # generated_text = tokenizer.tokenizer.decode(generated_ids)
                  input_seq_text = generated_text.split()[-len(input_seq):]
                  # input_seq_text = [ id_to_token[id] for id in generated_ids[-len(input_seq):] ]


                input_seq_ids = [token_to_id[token] if token in token_to_id else token_to_id['<UNK>'] for token in input_seq_text]
                input_seq = torch.tensor(input_seq_ids, dtype=torch.long).unsqueeze(0).to(device)
        generated_text = re.sub(r'\s+([.,;!?:\'])', r'\1', generated_text)
        return generated_text
