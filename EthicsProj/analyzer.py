# analyzer.py
import textstat
import numpy as np

def analyze_text_features(text):
    readability = textstat.flesch_reading_ease(text)
    sentence_count = textstat.sentence_count(text)
    word_count = textstat.lexicon_count(text)
    avg_sentence_length = word_count / max(1, sentence_count)
    lexical_diversity = len(set(text.split())) / max(1, word_count)
    return {
        "readability": readability,
        "sentence_count": sentence_count,
        "word_count": word_count,
        "avg_sentence_length": avg_sentence_length,
        "lexical_diversity": lexical_diversity
    }

def heuristic_score(features):
    read_score = 1 - (features["readability"] / 100)
    diversity_penalty = 1 - features["lexical_diversity"]
    length_factor = min(features["avg_sentence_length"] / 25, 1)
    return np.clip((read_score + diversity_penalty + length_factor) / 3, 0, 1)
