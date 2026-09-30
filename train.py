import os
import torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from datasets import load_dataset
from tqdm import tqdm

from model import ModelArgs, CustomLLM


class TextDataset(Dataset):
    """Датасет для подготовленных текстовых токенов."""
    def __init__(self, tokenizer_path: str, max_length: int = 512):
        self.tokenizer = Tokenizer.from_file(tokenizer_path)
        self.max_length = max_length
        
        print("Загрузка текста для обучения...")
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        
        print("Токенизация датасета...")
        full_text = "\n".join([t for t in dataset["text"] if len(t.strip()) > 0])
        encoded = self.tokenizer.encode(full_text)
        
        self.tokens = encoded.ids
        print(f"Всего токенов в обучающей выборке: {len(self.tokens):,}")

    def __len__(self):
        return (len(self.tokens) - 1) // self.max_length

    def __getitem__(self, idx):
        start_idx = idx * self.max_length
        end_idx = start_idx + self.max_length + 1
        
        chunk = self.tokens[start_idx:end_idx]
        
        # Если кусок короче нужного, дополняем токеном <pad> (0)
        if len(chunk) < self.max_length + 1:
            chunk = chunk + [0] * (self.max_length + 1 - len(chunk))
            
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def train():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Запуск обучения на устройстве: {device}")

    # 1. Параметры модели
    args = ModelArgs(
        dim=512,
        n_layers=6,
        n_heads=8,
        n_kv_heads=2,
        vocab_size=32768,
        max_seq_len=512
    )

    # 2. Инициализация модели
    model = CustomLLM(args).to(device)
    print(f"Параметров в модели: {sum(p.numel() for p in model.parameters()):,}")

    # 3. Подготовка данных
    dataset = TextDataset("tokenizer/tokenizer.json", max_length=args.max_seq_len)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # 4. Оптимизатор
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)

    # 5. Цикл обучения
    epochs = 3
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f"Эпоха {epoch + 1}/{epochs}")
        
        for x, y in progress_bar:
            x, y = x.to(device), y.to(device)
            
            optimizer.zero_grad()
            logits, loss = model(x, y)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Save model weights
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/model.pt")
    print("✅ Обучение завершено! Модель сохранена в 'checkpoints/model.pt'")


if __name__ == "__main__":
    train()
