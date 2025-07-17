import torch
from model import CharLSTM
from preprocess import importData, buildVocab, getPath
import re

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

def generate_line(model, char2idx, idx2char, start_text, max_length=300, temperature=1.0, device='cpu', stop_count=1):
   
    STOP_TOKENS = ['。', '！', '？', ',', '，']

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

def generate_poem(model, char2idx, idx2char, start_text, lines=4, temperature=1.0, device='cpu'):
    PUNC = r'，。！？，!?'
    def clean_len(s):
        return len(s.rstrip(PUNC))
    
    def is_valid_line(s, expected_len, allow_4plus4=False):
        stripped_len = clean_len(s)
        if stripped_len == expected_len:
            return True
        if allow_4plus4 and stripped_len == 8:
            comma_positions = [i for i, c in enumerate(s) if c in ['，', ',']]
            return 3 in comma_positions
        return False
    
    def strip_generated(s):
        matches = re.findall(rf'[^ {PUNC}]*[{PUNC}]', s)
        return matches[-1].strip() if matches else s
    
    first_line = generate_line(model, char2idx, idx2char, start_text, temperature=temperature, device=device)

    if not first_line:
        return "Failed to generate first line"
    
    first_line_clean_len = clean_len(first_line)

    print("First line is: " + first_line)

    if first_line_clean_len not in (5, 7):
        return "Failed to generate first line"
    
    poem_lines = [first_line]

    for i in range(lines - 1):
        next_line = None
        count = 0
        max_count = 5
        while count <= max_count:
            next_line = generate_line(model, char2idx, idx2char, start_text=poem_lines[i], temperature=temperature, device=device)
            next_line = strip_generated(next_line)
            if not next_line:
                continue  
            if first_line_clean_len == 5:
                if is_valid_line(next_line, expected_len=5):
                    break 
            elif first_line_clean_len == 7:
                if is_valid_line(next_line, expected_len=7, allow_4plus4=True):
                    break 
            count += 1
            # if failed just accept the last generated line
        poem_lines.append(next_line)
    return "\n".join(poem_lines)


def generate_acrostic(model, char2idx, idx2char, acroustic_phrase, temperature=1.0, device='cpu'):
    lines = []
    for c in acroustic_phrase:
        line = generate_line(model, char2idx, idx2char, c, stop_count=1, temperature=temperature, device=device)
        lines.append(line)
    return '\n'.join(lines)

if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    poems = importData(getPath("poetry.txt"))
    
    model_path = getPath(f'model{len(poems)}.pth')

    vocab, char2idx, idx2char = buildVocab(poems)

    model = load_model(model_path=model_path, vocab=vocab, embed_dim=128, hidden_dim=256, num_layers=2, device=device) 

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
        result = generate_poem(model, char2idx, idx2char, start_text=i, temperature=0.8, device=device)

        print(result)
    
    print("\nGenerated Acrostic:\n")
    
    result = generate_acrostic(model, char2idx, idx2char, acroustic_phrase=acrostic_prompt, device=device)

    print(result)
