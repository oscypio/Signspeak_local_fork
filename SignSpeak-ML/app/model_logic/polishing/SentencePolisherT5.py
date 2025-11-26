from transformers import T5ForConditionalGeneration, T5Tokenizer
import torch


class T5Polisher:
    """
    Polishes an already assembled gloss sentence (string)
    into natural English using FLAN-T5-small (~77MB).
    """

    def __init__(
        self,
        model_name: str = "google/flan-t5-small",
        device: str = "cpu"
    ):
        print(f"[SentencePolisher] Loading model: {model_name} on {device}")
        self.device = device

        self.tokenizer = T5Tokenizer.from_pretrained(model_name)
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        self.model.to(device)
        self.model.eval()

        print("[SentencePolisher] Ready.\n")

    # ---------------------------------------------------------------
    # Optional helper: remove repeated consecutive words
    # ---------------------------------------------------------------
    @staticmethod
    def remove_consecutive_duplicates(sentence: str) -> str:
        """
        Removes duplicated consecutive words:
        'HELLO HELLO NEED PHONE' → 'HELLO NEED PHONE'
        """
        words = sentence.split()
        cleaned = []

        for w in words:
            if not cleaned or cleaned[-1] != w:
                cleaned.append(w)

        return " ".join(cleaned)

    # ---------------------------------------------------------------
    # Main polishing function
    # ---------------------------------------------------------------
    def polish(self, gloss_sentence: str) -> str:
        """
        Takes a full gloss sentence (string),
        cleans duplicates, and rewrites into natural English.
        """
        if not gloss_sentence:
            return ""

        # 1) Clean duplicates
        cleaned = self.remove_consecutive_duplicates(gloss_sentence.strip())

        # 2) Build safe controlled prompt
        prompt = (
            "Rewrite this ASL gloss into ONE natural English sentence. "
            "Do NOT add new information. "
            "Do NOT explain anything. "
            "Do NOT repeat the input. "
            "Do NOT output multiple versions.\n\n"
            f"ASL gloss: {cleaned}"
        )

        # 3) Encode
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            max_length=128,
            truncation=True
        ).to(self.device)

        # 4) Generate corrected sentence
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )

        polished = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        ).strip()

        polished = polished[:1].upper() + polished[1:].lower()

        return polished
