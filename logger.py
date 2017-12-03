import torch
import numpy as np
from visdom import Visdom


class VisdomWriter(object):
    def __init__(self, title, xlabel='Epoch', ylabel='Loss'):
        """Extended Visdom Writer"""
        self.vis = Visdom()
        assert self.vis.check_connection()
        self.title = title
        self.xlabel = xlabel
        self.ylabel = ylabel
        self.x = 0
        self.win = None

    def update_text(self, text):
        """Text Memo (usually used to note hyperparameter-configurations)"""
        self.vis.text(text)

    def update(self, y):
        """Update loss (X: Step (Epoch) / Y: loss)"""
        self.x += 1
        if self.win is None:
            self.win = self.vis.line(
                X=np.array([self.x]),
                Y=np.array([y]),
                opts=dict(
                    title=self.title,
                    xlabel=self.xlabel,
                    ylabel=self.ylabel,
                ))
        else:
            self.vis.updateTrace(
                X=np.array([self.x]),
                Y=np.array([y]),
                win=self.win)


def log_samples(file_path, samples, EN, is_output=True):
    if is_output:
        _, argmax = torch.max(samples, 2)
        samples = argmax.cpu().data
    samples = samples.t()
    decoded_samples = []
    for i in range(len(samples)):
        decoded = ' '.join([EN.vocab.itos[s] for s in samples[i]])
        decoded_samples.append(decoded)
    with open(file_path, 'w+') as f:
        for sample in decoded_samples:
            f.write(sample + '\n')
