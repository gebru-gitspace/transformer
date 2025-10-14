# -*- coding: utf-8 -*-
from transformers import GPT2TokenizerFast

def load_tokenizer(tokenizer_path):
    tokenizer = GPT2TokenizerFast.from_pretrained(tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("Tokenizer loaded")
    print("Vocab size:", tokenizer.vocab_size)
    print("Special tokens:", tokenizer.all_special_tokens)
    print("-" * 50)
    return tokenizer

def test_single_sentences(tokenizer, sentences):
    for text in sentences:
        tokens = tokenizer.tokenize(text)
        ids = tokenizer.encode(text)
        decoded = tokenizer.decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        print(f"Input: {text}")
        print("Tokens:", tokens)
        print("IDs:", ids)
        print("Decoded:", decoded)
        print("-" * 50)

def test_batch(tokenizer, sentences):
    batch = tokenizer(sentences, padding=True, truncation=True, return_tensors="pt")
    print("Batch Input IDs shape:", batch["input_ids"].shape)
    print("Batch encoding done")
    print("-" * 50)

def main():
    # --- Config ---
    tokenizer_path = "./tokenizers/Tig_unigram_16000"
    test_sentences = [
        "ሰላም ሓውኻ ከመይ ኣለኻ?",
        "ኣነ ተምሃሮ እየ።",
        "ዝተማሃርኩ ፕሮግራሚንግ እዩ።"
    ]

    tokenizer = load_tokenizer(tokenizer_path)
    test_single_sentences(tokenizer, test_sentences)
    test_batch(tokenizer, test_sentences)

if __name__ == "__main__":
    main()
