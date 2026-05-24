from datasets import load_dataset
from transformers import AutoTokenizer

# Load dataset
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Get sample text
text = dataset["train"][5]["text"]

print("\nORIGINAL TEXT:\n")
print(text)

# Convert text into token IDs
tokens = tokenizer.encode(text)

print("\nTOKEN IDS:\n")
print(tokens)

print("\nNUMBER OF TOKENS:\n")
print(len(tokens))

# Define context window size
context_length = 8

print("\nCONTEXT WINDOWS:\n")

# Create chunks
for i in range(0, len(tokens) - context_length):

    x = tokens[i:i + context_length]
    y = tokens[i + 1:i + context_length + 1]

    print(f"\nINPUT  : {x}")
    print(f"TARGET : {y}")