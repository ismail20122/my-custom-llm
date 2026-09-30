import os
from datasets import load_dataset
from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers

def train_tokenizer():
    print("1. Загрузка обучающего датасета...")
    # Используем качественный открытый датасет Wikitext (или любой другой)
    dataset = load_dataset("wikitext", "wikitext-103-raw-v1", split="train")

    def batch_iterator(batch_size=1000):
        for i in range(0, len(dataset), batch_size):
            yield dataset[i : i + batch_size]["text"]

    print("2. Инициализация BPE токенизатора...")
    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=32768,
        special_tokens=["<unk>", "<s>", "</s>", "<pad>", "<mask_token>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )

    print("3. Обучение токенизатора на текстах...")
    tokenizer.train_from_iterator(batch_iterator(), trainer=trainer)

    # Создаем папку для сохранения результатов
    os.makedirs("tokenizer", exist_ok=True)
    tokenizer.save("tokenizer/tokenizer.json")
    print("✅ Токенизатор успешно обучен и сохранен в папку 'tokenizer/'!")

if __name__ == "__main__":
    train_tokenizer()
