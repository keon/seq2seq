# mini seq2seq
Minimal Seq2Seq model with attention for neural machine translation in PyTorch.

This implementation focuses on the following features:

- Modular structure to be used in other projects
- Minimal code for readability
- Full utilization of batches and GPU.

This implementation relies on [torchtext](https://github.com/pytorch/text) to minimize dataset management and preprocessing parts.

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


## References

Based on the following implementations

* [PyTorch Tutorial](http://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html)
* [@spro/practical-pytorch](https://github.com/spro/practical-pytorch)
* [@AuCson/PyTorch-Batch-Attention-Seq2seq](https://github.com/AuCson/PyTorch-Batch-Attention-Seq2seq)
