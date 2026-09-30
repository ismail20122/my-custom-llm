import os
import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tqdm import tqdm

def prepare_data():
    tokenizer_path = "tokenizer/tokenizer.json"
    if not os.path.exists(tokenizer_path):
        raise FileNotFoundError("Сначала нужно обучить токенизатор!")

    tokenizer = Tokenizer.from_file(tokenizer_path)
    
    print("1. Скачиваем тексты для обучения...")
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    print("2. Превращаем текст в токены...")
    all_tokens = []
    for text in tqdm(dataset["text"]):
        if text.strip():
            tokens = tokenizer.encode(text).ids
            all_tokens.extend(tokens)

    tokens_np = np.array(all_tokens, dtype=np.uint16)

    # 90% данных на обучение, 10% на проверку
    split_idx = int(len(tokens_np) * 0.9)
    train_data = tokens_np[:split_idx]
    val_data = tokens_np[split_idx:]

    os.makedirs("data", exist_ok=True)
    train_data.tofile("data/train.bin")
    val_data.tofile("data/val.bin")
    print("✅ Датасет успешно переведён в бинарный формат в папку 'data/'!")

if __name__ == "__main__":
    prepare_data()
