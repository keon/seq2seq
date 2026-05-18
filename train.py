import os
import math
import argparse
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.nn import functional as F
from model import Encoder, Decoder, Seq2Seq
from utils import load_dataset, PAD


def parse_arguments():
    p = argparse.ArgumentParser(description='Hyperparams')
    p.add_argument('-epochs', type=int, default=100)
    p.add_argument('-batch_size', type=int, default=32)
    p.add_argument('-lr', type=float, default=1e-4)
    p.add_argument('-grad_clip', type=float, default=10.0)
    p.add_argument('-hidden_size', type=int, default=512)
    p.add_argument('-embed_size', type=int, default=256)
    return p.parse_args()


def get_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def step_loss(model, src, trg, vocab_size, teacher_forcing_ratio):
    output = model(src, trg, teacher_forcing_ratio=teacher_forcing_ratio)
    return F.nll_loss(output[1:].reshape(-1, vocab_size),
                      trg[1:].reshape(-1), ignore_index=PAD)


def evaluate(model, val_iter, vocab_size, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for src, trg in val_iter:
            src, trg = src.to(device), trg.to(device)
            total_loss += step_loss(model, src, trg, vocab_size, 0.0).item()
    return total_loss / len(val_iter)


def train(model, optimizer, train_iter, vocab_size, grad_clip, device):
    model.train()
    total_loss = 0
    for b, (src, trg) in enumerate(train_iter):
        src, trg = src.to(device), trg.to(device)
        optimizer.zero_grad()
        loss = step_loss(model, src, trg, vocab_size, 0.5)
        loss.backward()
        clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total_loss += loss.item()
        if b % 100 == 0 and b != 0:
            print("[%d][loss:%5.2f][pp:%5.2f]" %
                  (b, total_loss / 100, math.exp(total_loss / 100)))
            total_loss = 0


def main():
    args = parse_arguments()
    device = get_device()
    print(f"[!] device: {device}")

    print("[!] preparing dataset...")
    train_iter, val_iter, test_iter, DE, EN = load_dataset(args.batch_size)
    de_size, en_size = len(DE), len(EN)
    print(f"[TRAIN]:{len(train_iter)}\t[TEST]:{len(test_iter)}")
    print(f"[DE_vocab]:{de_size} [EN_vocab]:{en_size}")

    print("[!] Instantiating models...")
    encoder = Encoder(de_size, args.embed_size, args.hidden_size,
                      n_layers=2, dropout=0.5)
    decoder = Decoder(args.embed_size, args.hidden_size, en_size,
                      n_layers=1, dropout=0.5)
    seq2seq = Seq2Seq(encoder, decoder).to(device)
    optimizer = optim.Adam(seq2seq.parameters(), lr=args.lr)
    print(seq2seq)

    best_val_loss = None
    for e in range(1, args.epochs + 1):
        train(seq2seq, optimizer, train_iter, en_size, args.grad_clip, device)
        val_loss = evaluate(seq2seq, val_iter, en_size, device)
        print("[Epoch:%d] val_loss:%5.3f | val_pp:%5.2f"
              % (e, val_loss, math.exp(val_loss)))
        if best_val_loss is None or val_loss < best_val_loss:
            print("[!] saving model...")
            os.makedirs(".save", exist_ok=True)
            torch.save(seq2seq.state_dict(), f'./.save/seq2seq_{e}.pt')
            best_val_loss = val_loss
    test_loss = evaluate(seq2seq, test_iter, en_size, device)
    print("[TEST] loss:%5.2f" % test_loss)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt as e:
        print("[STOP]", e)
