import torch
from model import CharLSTM
from preprocess import importData, buildVocab, getPath

def load_model(model_path, vocab, embed_dim, hidden_dim, num_layers, device):

    model = CharLSTM(len(vocab), embed_dim, hidden_dim, num_layers)

    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

    model.to(device)

    model.eval()

    return model

def get_model():
    poems = importData(getPath("poetry.txt"))
    vocab, char2idx, idx2char = buildVocab(poems)
    device = torch.device("cpu")
    model = load_model(
        model_path=getPath(f"model{len(poems)}.pth"),
        vocab=vocab,
        embed_dim=128,
        hidden_dim=256,
        num_layers=2,
        device=device
    )
    return model, char2idx, idx2char

def generate_text(model, char2idx, idx2char, start_text, max_length=300, temperature=1.0, device='cpu', stop_count=5):
   
   STOP_TOKENS = ['。', '！', '？']

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

               if stop_counter >= stop_count:
                   break
               
           input_seq = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)

   return generated

def generate_acrostic(model, char2idx, idx2char, acroustic_phrase, temperature=1.0, device='cpu'):
    lines = []
    for c in acroustic_phrase:
        line = generate_text(model, char2idx, idx2char, c, stop_count=1, temperature=temperature, device=device)
        lines.append(line)
    return '\n'.join(lines)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    poems = importData(getPath("poetry.txt"))
    
    model_path = getPath(f'model{len(poems)}.pth')

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

    acrostic_prompt = "我在草原上"

    print("\nGenerated Poems:\n")
    
    for i in prompts:
        result = generate_text(model, char2idx, idx2char, start_text=i, temperature=0.8, device=device, stop_count=5)

        print(result)
    
    print("\nGenerated Acrostic:\n")
    
    result = generate_acrostic(model, char2idx, idx2char, acroustic_phrase=acrostic_prompt, device=device)

    print(result)
