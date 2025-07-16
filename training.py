from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import CharLSTM
from preprocess import importData, buildVocab, PoemDataset, collate, PAD_IDX, getPath

def train():
    num_epochs = 999
    embed_dim = 128
    hidden_dim = 256 
    num_layers = 2
    learning_rate = 0.003

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(device)

    poem_path = getPath('poetry.txt')

    poems = importData(poem_path)

    model_path = getPath(f'model{len(poems)}.pth')

    vocab, char2idx, _ = buildVocab(poems)

    dataset = PoemDataset(poems, char2idx)
    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=lambda b: collate(b, pad_idx=PAD_IDX))

    model = CharLSTM(len(vocab), embed_dim, hidden_dim, num_layers).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loading existing model {model_path}")
    except FileNotFoundError:
        print(f"Creating new model {model_path}")

    criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in range(num_epochs):

        total_loss = 0

        model.train()

        for x, y in tqdm(dataloader, desc="Training", unit="batch"):

            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            output, _ = model(x)

            loss = criterion(output.view(-1, len(vocab)), y.view(-1))

            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)

            optimizer.step()

            total_loss += loss.item()
          
        avg_loss = total_loss / len(dataloader)

        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")
        with open(getPath("loss_log.txt"), "a", encoding="utf-8") as f:
            f.write(f"{epoch+1},{avg_loss:.4f}\n")

        print("Saving data...")
        torch.save(model.state_dict(), model_path)
        print("Data saved!")
    
    torch.save(model.state_dict(), model_path)
    print("Training complete")

if __name__ == '__main__':
    try:
        train()
    except KeyboardInterrupt:
        print("Interrupting training.")