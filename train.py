import os
import math
import argparse
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from utils import load_dataset
from logger import VisdomWriter, log_samples


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100,
                   help='number of epochs for train')
    p.add_argument('-batch_size', type=int, default=64,
                   help='number of epochs for train')
    p.add_argument('-lr', type=float, default=0.0001,
                   help='initial learning rate')
    p.add_argument('-grad_clip', type=float, default=10.0,
                   help='initial learning rate')
    return p.parse_args()


def evaluate(model, val_iter, vocab_size, DE, EN):
    model.eval()
    pad = EN.vocab.stoi['<pad>']
    total_loss = 0
    for b, batch in enumerate(val_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.cuda(), trg.cuda()
        output = model(src, trg)
        loss = F.cross_entropy(output.view(-1, vocab_size),
                               trg.contiguous().view(-1), ignore_index=pad)
        total_loss += loss.data[0]
    return total_loss / len(val_iter)


def train(e, model, optimizer, train_iter, vocab_size, grad_clip, vis, DE, EN):
    model.train()
    losses = 0
    pad = EN.vocab.stoi['<pad>']
    for b, batch in enumerate(train_iter):
        src, len_src = batch.src
        trg, len_trg = batch.trg
        src, trg = src.cuda(), trg.cuda()
        optimizer.zero_grad()
        output = model(src, trg)
        loss = F.cross_entropy(output.view(-1, vocab_size),
                               trg.contiguous().view(-1), ignore_index=pad)
        loss.backward()
        clip_grad_norm(model.parameters(), grad_clip)
        optimizer.step()
        loss = loss.data[0]
        losses += loss

        if b % 100 == 0 and b != 0:
            losses = losses / 100
            print("[%d][loss:%5.2f][pp:%5.2f]" % (b, losses, math.exp(losses)))
            vis.update(loss)
            losses = 0
    if not os.path.isdir(".samples"):
        os.makedirs(".samples")
    log_samples('./.samples/de-%d.txt' % e, src.data, DE, is_sample=False)
    log_samples('./.samples/en-%d.txt' % e, output, EN)


def main():
    args = parse_arguments()
    hidden_size = 1024
    embed_size = 512
    assert torch.cuda.is_available()

    # visdom for plotting
    vis_train = VisdomWriter("Training Loss", xlabel='Batch', ylabel='Loss')
    vis_val = VisdomWriter("Validation Loss", xlabel='Epoch', ylabel='Loss')

    print("[!] preparing dataset...")
    train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size)
    de_size, en_size = len(DE.vocab), len(EN.vocab)
    print("de_vocab_size: %d en_vocab_size: %d" % (de_size, en_size))

    print("[!] Instantiating models...")
    encoder = Encoder(de_size, embed_size, hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(embed_size, hidden_size, en_size,
                      n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).cuda()
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    best_val_loss = None
    for e in range(1, args.epochs+1):
        train(e, seq2seq, optimizer, train_iter,
              en_size, args.grad_clip, vis_train, DE, EN)
        val_loss = evaluate(seq2seq, val_iter, en_size, DE, EN)
        vis_val.update(val_loss)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2fS"
              % (e, val_loss, math.exp(val_loss)))

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            print("[!] saving model...")
            if not os.path.isdir(".save"):
                os.makedirs(".save")
            torch.save(seq2seq.state_dict(), './.save/seq2seq_%d.pt' % (e))
            best_val_loss = val_loss


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
