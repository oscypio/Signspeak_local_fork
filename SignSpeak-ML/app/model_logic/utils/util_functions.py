from .config import settings


def generate_no_word_response():
    """
    Returns no word found response in API-like schema
    :return: Api-like schema response
    """
    return {
        "prediction": None,
        "status": "error",
        "current_words": '',
        "detail": "Empty or invalid landmark sequence"
    }

def generate_given_word_response(word: str, current_words : list, confidence: float = None):
    """
    Returns word added response in API-like schema
    :param word: Detected word
    :param current_words: List of accumulated words
    :param confidence: Confidence score (0.0 - 1.0), optional
    :return: Api-like schema response
    """
    response = {
        "prediction": word,
        "status": "word_added",
        "current_words": current_words,
        "detail": "Word was added successfully"
    }

    # Add confidence if provided
    if confidence is not None:
        response["confidence"] = float(confidence)

    return response

def generate_end_of_sentence_response(sentence : str, word: str = settings.SPECIAL_LABEL):
    """
    Returns no word found response in API-like schema
    :return: Api-like schema response
    """
    return {
        "prediction": word,
        "status": "end_of_sentence",
        "sentence": sentence,
        "detail": "Sentence formed"
    }
