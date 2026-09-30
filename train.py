import os
import time
import math
import numpy as np
import torch
from model import CustomLLM, ModelArgs

def get_batch(data, block_size, batch_size, device):
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Запуск обучения на устройстве: {device}")

    train_data = np.memmap("data/train.bin", dtype=np.uint16, mode="r")
    val_data = np.memmap("data/val.bin", dtype=np.uint16, mode="r")

    # Конфигурация нашей модели
    args = ModelArgs(
        dim=384,
        n_layers=6,
        n_heads=6,
        n_kv_heads=2,
        vocab_size=32768,
        max_seq_len=256
    )

    model = CustomLLM(args).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=6e-4)

    batch_size = 16
    block_size = 256
    max_iters = 1000

    print("Старт цикла обучения...")
    model.train()
    for step in range(1, max_iters + 1):
        X, Y = get_batch(train_data, block_size, batch_size, device)
        
        optimizer.zero_grad()
        _, loss = model(X, Y)
        loss.backward()
        optimizer.step()

        if step % 50 == 0:
            print(f"Шаг {step}/{max_iters} | Loss (ошибка): {loss.item():.4f}")

    # Сохраняем готовую модель
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.state_dict(), "checkpoints/my_model.pt")
    print("🎉 Обучение завершено! Модель сохранена в 'checkpoints/my_model.pt'")

if __name__ == "__main__":
    main()
