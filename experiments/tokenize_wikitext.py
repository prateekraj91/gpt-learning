from datasets import load_dataset
from transformers import AutoTokenizer

# Load WikiText-2 dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Print dataset info
print(dataset)

# Load GPT-2 tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Try custom text instead of dataset text
sample_text = "I want to build my own GPT model."

# Print original text
print("\nSAMPLE TEXT:\n")
print(sample_text)

# Show how tokenizer splits words/subwords
print("\nTOKENIZED WORDS:\n")
print(tokenizer.tokenize(sample_text))

# Convert text into tokens
tokens = tokenizer(sample_text)

# Print full tokenized output
print("\nTOKENIZED OUTPUT:\n")
print(tokens)

# Print token IDs only
print("\nTOKEN IDS:\n")
print(tokens["input_ids"])

# Decode tokens back into text
decoded = tokenizer.decode(tokens["input_ids"])

print("\nDECODED TEXT:\n")
print(decoded)