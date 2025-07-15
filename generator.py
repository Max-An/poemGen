import os
import torch
from model import CharLSTM
from preprocess import importData, buildVocab


def load_model(model_path, vocab, embed_dim, hidden_dim, num_layers, device):

    model = CharLSTM(len(vocab), embed_dim, hidden_dim, num_layers)

    model.load_state_dict(torch.load(model_path, map_location=device))

    model.to(device)

    model.eval()

    return model

def generate_text(model, char2idx, idx2char, start_text, max_length=300, temperature=1.0, device='cpu'):
   
   STOP_TOKENS = ['。', '！', '？']

   STOP_COUNT = 5
   
   input_seq = torch.tensor(
       [[char2idx.get(c, char2idx["<UNK>"]) for c in start_text]],
       dtype=torch.long
   ).to(device)

   generated = start_text

   hidden = None

   stop_counter = 0

   with torch.no_grad():

       for _ in range(max_length):
           
           output, hidden = model(input_seq, hidden)

           last_logits = output[0, -1] / temperature

           probs = torch.softmax(last_logits, dim=0)

           next_char_idx = torch.multinomial(probs, 1).item()

           next_char = idx2char[next_char_idx]

           generated += next_char

           if next_char in STOP_TOKENS:
               stop_counter += 1

               if stop_counter >= STOP_COUNT:
                   break
               
           input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)

   return generated


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    poems = importData(os.path.join("poemGen", "poetry.txt"))
    
    model_path = os.path.join('poemGen', f'model{len(poems)}.pth')

    vocab, char2idx, idx2char = buildVocab(poems)

    model = load_model(

        model_path = model_path, 

        vocab=vocab,

        embed_dim=128,

        hidden_dim=256,

        num_layers=2,

        device=device

    ) 

    prompts = [
        "一",
        "如",
        "春", 
        "沙",
        "雨",
        "忽",
        "?",
    ]

    print("\nGenerated Poems:\n")
    
    for i in prompts:
        result = generate_text(model, char2idx, idx2char, start_text=i, temperature=0.8, device=device)

        print(result)
