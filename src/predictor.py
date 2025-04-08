from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from typing import List

class LLMPredictor:
    def __init__(self, model_name = "model/distilgpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

    def predict(self, input_word: str, candidate_words: List[str]) -> str:
        # Tokenize input text
        inputs = self.tokenizer(input_word, return_tensors="pt").to(self.model.device)

        # Get logits (raw predictions before softmax)
        with torch.no_grad():
            outputs = self.model(**inputs)

        logits = outputs.logits  # Shape: (batch_size, sequence_length, vocab_size)

        # Get the last token's logits
        last_token_logits = logits[0, -1, :]  # Only take the last word's probabilities

        # Convert logits to probabilities
        probabilities = torch.softmax(last_token_logits, dim=-1)

        # Get token IDs for the candidate words
        candidate_ids = {word: self.tokenizer.convert_tokens_to_ids(self.tokenizer.tokenize(word)[0]) for word in candidate_words}

        # Add the EOS token
        eos_token = self.tokenizer.eos_token  # This is "<|endoftext|>" for GPT-2
        eos_token_id = self.tokenizer.eos_token_id
        candidate_ids[eos_token] = eos_token_id

        # Extract probabilities for each candidate word + EOS token
        candidate_probs = {word: probabilities[idx].item() for word, idx in candidate_ids.items()}

        best_word = max(candidate_probs, key=candidate_probs.get)
        return best_word

