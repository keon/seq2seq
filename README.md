# mini seq2seq
Minimal Seq2Seq model with attention for neural machine translation in PyTorch.

This implementation focuses on the following features:

- Modular structure to be used in other projects
- Minimal code for readability
- Full utilization of batches and GPU.

Dataset (Multi30k DE→EN) is loaded via HuggingFace [`datasets`](https://github.com/huggingface/datasets); tokenization uses [spaCy](https://spacy.io/).

## Model description

* Encoder: Bidirectional GRU
* Decoder: GRU with Attention Mechanism
* Attention: [Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473)

![](http://www.wildml.com/wp-content/uploads/2015/12/Screen-Shot-2015-12-30-at-1.16.08-PM.png)

## Requirements

* Python 3.9+
* PyTorch >= 2.0 (CPU, CUDA, or Apple MPS)
* `datasets` (HuggingFace, replaces torchtext)
* Spacy >= 3.7

```
pip install -r requirements.txt
python -m spacy download de_core_news_sm
python -m spacy download en_core_web_sm
```

## Train

```
python train.py -epochs 30 -batch_size 32 -lr 3e-4
```

Device is auto-detected (CUDA → MPS → CPU). Smaller `-hidden_size` / `-embed_size` flags are useful for CPU smoke runs.

Sanity check (CPU, 500 batches, hidden=128/embed=64):

| step | train loss | perplexity |
|------|-----------:|-----------:|
| init |       9.20 |      9930 |
|   50 |       6.95 |      1045 |
|  100 |       5.46 |       236 |
|  250 |       5.16 |       175 |
|  500 |       4.89 |       133 |

Final val loss: **4.97** (random-init prior is `log(|V|) ≈ 9.19`).

## References

Based on the following implementations

* [PyTorch Tutorial](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
* [@spro/practical-pytorch](https://github.com/spro/practical-pytorch)
* [@AuCson/PyTorch-Batch-Attention-Seq2seq](https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq)
