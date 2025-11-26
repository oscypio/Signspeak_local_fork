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

def generate_given_word_response(word: str, current_words : list):
    """
    Returns no word found response in API-like schema
    :return: Api-like schema response
    """
    return {
        "prediction": word,
        "status": "word_added",
        "current_words": current_words,
        "detail": "Word was added successfully"
    }

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
