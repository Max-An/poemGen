import os
import sys
import torch
from torch.utils.data import Dataset
from collections import Counter
from torch.nn.utils.rnn import pad_sequence

PAD_CHAR = "<PAD>"
UNK_CHAR = "<UNK>"
PAD_IDX = 0
UNK_IDX = 1
BATCH_SIZE = 64
MAX_LEN = 150

def resourcePath(relative_path):

    if hasattr(sys, '_MEIPASS'):

        return os.path.join(sys._MEIPASS, relative_path)

    return os.path.join(os.path.dirname(__file__), relative_path)
 
DATA_DIR = resourcePath("poemGen")

def getPath(filename): 
    return os.path.join(DATA_DIR, filename)

def importData(path):
    poems = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if ':' in line:
                _, poem = line.split(':', 1)
            else:
                poem = line
            poems.append(poem)
    return poems

def buildVocab(poems):
    text = ''.join(poems)
    counter = Counter(text)
    sorted_chars = [char for char, _ in counter.most_common()]
    vocab = [PAD_CHAR, UNK_CHAR] + sorted_chars
    char2idx = {char: idx for idx, char in enumerate(vocab)}
    idx2char = {idx: char for char, idx in char2idx.items()}
    return vocab, char2idx, idx2char

class PoemDataset(Dataset):
   def __init__(self, poems, char2idx):
       self.char2idx = char2idx
       self.unk_idx = char2idx[UNK_CHAR]
       self.data = []
       for poem in poems:
           encoded = [char2idx.get(c, self.unk_idx) for c in poem]
           if 2 <= len(encoded) <= MAX_LEN:
               x = encoded[:-1]
               y = encoded[1:]
               self.data.append((x, y))

   def __len__(self):
       return len(self.data)
   
   def __getitem__(self, idx):
       return self.data[idx]
 
def collate(batch, pad_idx=0):
    xs, ys = zip(*batch)
    xs = [torch.tensor(seq, dtype=torch.long) for seq in xs]
    ys = [torch.tensor(seq, dtype=torch.long) for seq in ys]
    x_padded = pad_sequence(xs, batch_first=True, padding_value=pad_idx)
    y_padded = pad_sequence(ys, batch_first=True, padding_value=pad_idx)
    return x_padded, y_padded