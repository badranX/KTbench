import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import functools

def padder(seqs, out_maxlen=-1, padding_value=0):
    lens = torch.tensor(list(map(len, seqs)), dtype=int)
    maxlen = lens.max()
    idxs = None
    if out_maxlen > 0:
        if maxlen <= out_maxlen:
            seqs[0] = F.pad(seqs[0], (0, maxlen - seqs[0].shape[0]), "constant", padding_value)
        else:
            #this won't be reached in training
            nsplits = -(-lens//out_maxlen)
            nsplits = nsplits.tolist()
            def split_seq(seq, num):
                trail = len(seq) - out_maxlen*(num-1)
                return list(seq.split([out_maxlen]*(num-1) + [trail]))

            tmp = map(lambda x: split_seq(*x), zip(seqs, nsplits))

            seqs = functools.reduce(lambda x,y: x + y, tmp, [])
            idxs = functools.reduce(lambda x,y: x + [y]*y, nsplits, [])
            lens = torch.tensor(list(map(len, seqs)), dtype=int)

    seq = pad_sequence(seqs, batch_first=True, padding_value=padding_value)
    return seq, lens, idxs

def padder_list(seqs, out_maxlen=-1, padding_value=0, dtype=float, to_tensor=False):
    seqs = list(map(lambda x: torch.tensor(x, dtype=dtype), seqs))
    seq, lens, idxs = padder(seqs, out_maxlen, padding_value)
    if to_tensor:
        return seq, lens, idxs
    else:
        return seq.tolist(), lens.tolist(), idxs


def splitter(seqs, out_maxlen):
    lens = torch.tensor(list(map(len, seqs)), dtype=int)
    maxlen = lens.max()
    idxs = None
    if maxlen <= out_maxlen:
        return seqs, lens, None
    else:
        #this won't be reached in training
        nsplits = -(-lens//out_maxlen)
        nsplits = nsplits.tolist()
        def split_seq(seq, num):
            trail = len(seq) - out_maxlen*(num-1)
            return list(seq.split([out_maxlen]*(num-1) + [trail]))

        tmp = map(lambda x: split_seq(*x), zip(seqs, nsplits))

        seqs = functools.reduce(lambda x,y: x + y, tmp, [])
        idxs = functools.reduce(lambda x,y: x + [y]*y, nsplits, [])
        lens = torch.tensor(list(map(len, seqs)), dtype=int)

    return seqs, lens, idxs

if __name__ == "__main__":
    # Desired max length
    max_len = 50

    # 100 seqs of variable length (< max_len)
    seq_lens = torch.randint(low=10,high=44,size=(100,))
    seqs = [torch.rand(n) for n in seq_lens]


    # pad all seqs to desired length
    seqs = padder(seqs, 70)

    print(seqs)
