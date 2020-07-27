import argparse
import random
import torch

import numpy as np

from bert_pytorch import parse_args
from bert_pytorch.trainer import BERTTrainer
from bert_pytorch.dataset import BERTDataset, WordVocab
from bert_pytorch.model import BERT
from torch.utils.data import DataLoader

torch.manual_seed(1337)
random.seed(1337)
np.random.seed(1337)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

def skipIfNotImplemented(func):
    def wrapper(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except NotImplementedError:
            print('skipped since {} is not implemented'.format(func.__name__))
    return wrapper

class Model:
    def __init__(self, device=None, jit=False):
        self.device = device
        self.jit = jit


    def get_module(self):
        args = parse_args(args=[
            '--train_dataset', 'data/corpus.small',
            '--test_dataset', 'data/corpus.small',
            '--vocab_path', 'data/vocab.small',
            '--output_path', 'bert.model',
        ]) # Avoid reading sys.argv here
        print("Loading Vocab", args.vocab_path)
        vocab = WordVocab.load_vocab(args.vocab_path)
        print("Vocab Size: ", len(vocab))

        train_dataset = BERTDataset(args.train_dataset, vocab, seq_len=args.seq_len,
                                    corpus_lines=args.corpus_lines, on_memory=args.on_memory)
        test_dataset = BERTDataset(args.test_dataset, vocab, seq_len=args.seq_len, on_memory=args.on_memory) \
            if args.test_dataset is not None else None

        print("Creating Dataloader")
        train_data_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers)
        test_data_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=args.num_workers) \
            if test_dataset is not None else None

        print("Building BERT model")
        bert = BERT(len(vocab), hidden=args.hidden, n_layers=args.layers, attn_heads=args.attn_heads)

        if self.jit:
            print("Scripting BERT model")
            bert = torch.jit.script(bert)

        trainer = BERTTrainer(bert, len(vocab), train_dataloader=train_data_loader, test_dataloader=test_data_loader,
                            lr=args.lr, betas=(args.adam_beta1, args.adam_beta2), weight_decay=args.adam_weight_decay,
                            with_cuda=args.with_cuda, cuda_devices=args.cuda_devices, log_freq=args.log_freq, debug=args.debug)
        return trainer, None

    @skipIfNotImplemented
    def eval(self, niter=1):
        m, _ = self.get_module()
        for _ in range(niter):
            m.test(epoch=0)

    @skipIfNotImplemented
    def train(self, niter=1):
        m, _ = self.get_module()
        for _ in range(niter):
            m.train(epoch=0)


if __name__ == '__main__':
    m = Model(device='cuda', jit=False)
    model, _ = m.get_module()
    # model(*example_inputs)
    m.train()
    m.eval()
