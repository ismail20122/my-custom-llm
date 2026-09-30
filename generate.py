import torch
import torch.nn.functional as F
from tokenizers import Tokenizer
from model import ModelArgs, CustomLLM


def generate(
    prompt: str,
    model_path: str = "checkpoints/model.pt",
    tokenizer_path: str = "tokenizer/tokenizer.json",
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: int = 40
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = Tokenizer.from_file(tokenizer_path)

    # Конфигурация модели (должна совпадать с train.py)
    args = ModelArgs(
        dim=512,
        n_layers=6,
        n_heads=8,
        n_kv_heads=2,
        vocab_size=32768,
        max_seq_len=512
    )

    model = CustomLLM(args).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Токенизируем промпт
    tokens = tokenizer.encode(prompt).ids
    tokens_tensor = torch.tensor([tokens], dtype=torch.long, device=device)

    print(f"\n--- Промпт: {prompt} ---\n")
    print(prompt, end="", flush=True)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            idx_cond = tokens_tensor[:, -args.max_seq_len:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            tokens_tensor = torch.cat((tokens_tensor, next_token), dim=1)
            
            decoded_word = tokenizer.decode([next_token.item()])
            print(decoded_word, end="", flush=True)

    print("\n\n--- Конец генерации ---")


if __name__ == "__main__":
    generate("The future of artificial intelligence is")
