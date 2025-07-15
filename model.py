import torch
import torch.nn as nn

class CharLSTM(nn.Module):
   def __init__(self, vocab_size, embed_dim=128, hidden_dim=256, num_layers=2):
       super().__init__()
       self.embed = nn.Embedding(vocab_size, embed_dim)
       self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers, batch_first=True)
       self.fc = nn.Linear(hidden_dim, vocab_size)
       
   def forward(self, x, hidden=None):
       x = self.embed(x)  
       out, hidden = self.lstm(x, hidden)
       out = self.fc(out)
       return out, hidden