from llama_cpp import Llama
from ..utils.config import settings


class SentencePolisher:
    def __init__(self, model_path: str = settings.POLISHING_MODEL_PATH):
        print(f"Loading polishing model: {model_path}")
        self.llm = Llama(
            model_path=model_path,
            n_ctx=1024,
            n_threads=6,
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

        prompt = f"""
        Rewrite the ASL gloss into simple English.
        
        RULES:
        - Output exactly ONE short English sentence.
        - Do NOT add any information.
        - Do NOT list alternatives.
        - Do NOT explain anything.
        - Do NOT repeat the input.
        - Remove same words from sentence.
        - Do NOT output anything except the final sentence.
        - Try to make it sound natural and with sense.
        - Reorder words, so that the sentence have the most sense.
        - Add punctuation as needed.
        
        ASL gloss: {sentence}
        
        English:
        """

        output = self.llm(
            prompt,
            max_tokens=50,
            temperature=0.2,
            stop=["\n"]
        )

        text = output["choices"][0]["text"].strip()
        if "." in text:
            text = text.split(".")[0].strip() + "."

        text = text[:1].upper() + text[1:].lower()

        return text

