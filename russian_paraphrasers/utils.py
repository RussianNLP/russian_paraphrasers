import re
import nltk
import torch


def set_seed(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def adjust_length_to_model(length, max_sequence_length):
    if length < 0 and max_sequence_length > 0:
        length = max_sequence_length
    elif 0 < max_sequence_length < length:
        length = max_sequence_length  # No generation bigger than model size
    elif length < 0:
        length = int(10000)  # avoid infinite loop
    return length


def clean(text):
    if len(text) > 1:
        text = text.split("===")[0]
        text = text.split("\n")[0]
        text = re.sub('\\n', '\n', text)
        text = re.sub('<UNK>', '', text)
        text = re.sub('&amp;', '&', text)
        text = re.sub('lt;', '', text)
        text = re.sub('gt;', '', text)
        text = text.split("< EOS>")[0]
        text = text.split("<EOS>")[0]
        text = re.sub('< EOS>', ' ', text)
        text = re.sub('<s>', '', text)
        text = re.sub('</s>', '', text)
        text = re.sub('<EOS>', ' ', text)
        text = re.sub('< BOS>', ' ', text)
        text = re.sub('<BOS>', ' ', text)
        text = re.sub('< SHORT>', ' ', text)
        text = re.sub('<SHORT>', ' ', text)
        text = re.sub('<LONG>', ' ', text)
        text = re.sub('< LONG>', ' ', text)
        text = re.sub(' ul ', '\n', text)
        text = re.sub(' pre ', ' ', text)
        text = re.sub(r' /pre ', ' ', text)
        text = re.sub(r' / pre ', ' ', text)
        text = re.sub(r'/code', '\n/code\n', text)
        text = re.sub(r'/ code', '\n/code\n', text)
        text = re.sub(' code', '\ncode\n', text)
        text = re.sub(' hr ', ' ', text)
        text = re.sub(' e f ', '\n', text)
        text = re.sub('/h1', '\n', text)
        text = re.sub('nbsp;', ' ', text)
        text = re.sub('/blockquote', '\n', text)
        text = re.sub(' +', ' ', text)
        text = re.sub('&zwj;', '', text)
        text = re.sub('.<', '.', text)
        text = re.sub('/', '.', text)
        text = re.sub('tml', '', text)
        text = re.sub("</s", '', text)
        text = re.sub("..s", '', text)
        text = re.sub("&#[0-9]+;", "", text)
        text = text.replace("ћ", "м").replace("ƒ", "д")
        text = text.replace("(версия 2)", "").replace("(примечание)", "")
    return text.strip()
