import torch
import spacy
from collections import Counter
from datasets import load_dataset as _hf_load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

UNK, PAD, SOS, EOS = 0, 1, 2, 3
SPECIALS = ['<unk>', '<pad>', '<sos>', '<eos>']


class Vocab:
    def __init__(self, counter, min_freq=1, max_size=None):
        self.itos = list(SPECIALS) + [
            w for w, c in counter.most_common(max_size)
            if c >= min_freq and w not in SPECIALS
        ]
        self.stoi = {w: i for i, w in enumerate(self.itos)}

    def __len__(self):
        return len(self.itos)

    def encode(self, tokens):
        return [SOS] + [self.stoi.get(t, UNK) for t in tokens] + [EOS]


def load_dataset(batch_size, dataset_name='bentrevett/multi30k'):
    de_nlp = spacy.load('de_core_news_sm')
    en_nlp = spacy.load('en_core_web_sm')
    tok_de = lambda t: [w.text.lower() for w in de_nlp.tokenizer(t)]
    tok_en = lambda t: [w.text.lower() for w in en_nlp.tokenizer(t)]

    raw = _hf_load_dataset(dataset_name)
    train, val, test = raw['train'], raw['validation'], raw['test']

    de_c, en_c = Counter(), Counter()
    for ex in train:
        de_c.update(tok_de(ex['de']))
        en_c.update(tok_en(ex['en']))
    DE = Vocab(de_c, min_freq=2)
    EN = Vocab(en_c, max_size=10000)

    def collate(batch):
        src = [torch.tensor(DE.encode(tok_de(b['de']))) for b in batch]
        trg = [torch.tensor(EN.encode(tok_en(b['en']))) for b in batch]
        return (pad_sequence(src, padding_value=PAD),
                pad_sequence(trg, padding_value=PAD))

    def loader(split, shuffle):
        return DataLoader(split, batch_size=batch_size,
                          collate_fn=collate, shuffle=shuffle)
    return loader(train, True), loader(val, False), loader(test, False), DE, EN
