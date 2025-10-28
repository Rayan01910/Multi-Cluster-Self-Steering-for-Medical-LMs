import re
import string
from normalize_answer import normalize_answer


def get_answer(text):
    pattern = r"<answer>\s*(.*?)\s*</answer>"  # capture ignoring surrounding spaces
    match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
    if match:
        return match.group(1).strip()  # strip whitespace/newlines around extracted answer
    return ""


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))