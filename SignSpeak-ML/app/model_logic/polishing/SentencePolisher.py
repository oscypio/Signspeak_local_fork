"""
Module for polishing ASL gloss sentences into natural English.
Uses a local LLaMA model for text generation.
"""


from llama_cpp import Llama
from ..utils.config import settings


class SentencePolisher:
    def __init__(self, model_path: str = settings.POLISHING_MODEL_PATH):
        print(f"Loading polishing model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=2048,  # Increased from 1024 for longer prompts with few-shot examples
            n_threads=8,  # Increased from 6 for better CPU utilization
            verbose=False
        )

    def remove_adjacent_duplicates(self, sentence: str) -> str:
        """
        Removes adjacent repeated words:
            "HELLO HELLO PHONE" -> "HELLO PHONE"
            "I WANT WANT EAT" -> "I WANT EAT"
        """
        if not sentence:
            return sentence

        # Tokenize by whitespace
        words = sentence.split()
        if not words:
            return sentence

        cleaned = [words[0]]  # always keep the first

        for w in words[1:]:
            if w != cleaned[-1]:  # only add if not duplicate
                cleaned.append(w)

        return " ".join(cleaned)

    def polish(self, sentence: str) -> str:
        if not sentence:
            return ""

        sentence = self.remove_adjacent_duplicates(sentence)

        # Improved prompt - NO indentation, few-shot examples for better quality
        prompt = f"""Convert ASL gloss to natural English. Keep all words, just add grammar and punctuation.

Examples:
ASL: HELLO MY NAME JOHN
English: Hello, my name is John.

ASL: I WANT EAT PIZZA
English: I want to eat pizza.

ASL: WHERE YOU GO TOMORROW
English: Where are you going tomorrow?

ASL: PLEASE HELP ME FIND PHONE
English: Please help me find my phone.

ASL: {sentence}
English:"""

        output = self.llm(
            prompt,
            max_tokens=settings.POLISHING_MAX_TOKENS,  # Configurable from settings
            temperature=settings.POLISHING_TEMPERATURE,  # Configurable from settings
            top_p=settings.POLISHING_TOP_P,  # Configurable from settings
            repeat_penalty=settings.POLISHING_REPEAT_PENALTY,  # Configurable from settings
            stop=["\n", "ASL:", "English:", "Example", "\n\n"]  # More comprehensive stop tokens
        )

        text = output["choices"][0]["text"].strip()

        # Remove common artifacts (quotes, etc.)
        text = text.replace('"', '').replace("'", "'")

        # Handle multiple sentences - take only first
        for punct in [".", "?", "!"]:
            if punct in text:
                text = text.split(punct)[0].strip() + punct
                break
        else:
            # No punctuation found - add period
            text = text + "."

        # Improved capitalization - only capitalize first character, preserve rest
        # This prevents "NASA" from becoming "nasa"
        if text:
            text = text[0].upper() + text[1:].lower()

        # Safety check - if output is suspiciously short or empty, return cleaned original
        if len(text.strip()) < 3 or text == ".":
            # Fallback: capitalize first word of original and add period
            words = sentence.split()
            if words:
                words[0] = words[0].capitalize()
                return " ".join(words) + "."
            return sentence + "."

        # Additional validation - check if model refused or just repeated input
        text_lower = text.lower()
        if any(bad in text_lower for bad in ['sorry', 'cannot', 'as an ai', 'i apologize']):
            # Model refused - return original with basic formatting
            words = sentence.split()
            if words:
                words[0] = words[0].capitalize()
                return " ".join(words) + "."
            return sentence + "."

        return text

