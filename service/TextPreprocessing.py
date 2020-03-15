import re
import string


def clean_str(target):
    """
    Tokenization/string cleaning for all datasets.
    """
    target = re.sub(r"\'s", " \'s", target)
    target = re.sub(r"\'ve", " \'ve", target)
    target = re.sub(r"n\'t", " n\'t", target)
    target = re.sub(r"\'re", " \'re", target)
    target = re.sub(r"\'d", " \'d", target)
    target = re.sub(r"\'ll", " \'ll", target)
    target = re.sub(r",", " , ", target)
    target = re.sub(r"!", " ! ", target)
    target = re.sub(r"\(", " \( ", target)
    target = re.sub(r"\)", " \) ", target)
    target = re.sub(r"\?", " \? ", target)
    target = re.sub(r"\s{2,}", " ", target)
    target = re.sub(r"\s{2,}", " ", target)
    return target.translate(str.maketrans('', '', string.punctuation)).strip().lower()