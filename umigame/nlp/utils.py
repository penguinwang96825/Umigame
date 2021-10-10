import numpy as np
from itertools import starmap, repeat, chain, islice


def padding(seq, size, mode):
    """
    Parameters
    ----------
    seq: np.array
        The sequence to be padded.
    mode: str
        Select padding mode among {"zero", "repeat"}.
        
    Returns
    -------
    seq: np.ndarray
    """
    if mode == "zero":
        seq = np.array(trimmer(seq, size, filler=0))
    elif mode == "repeat":
        seq = np.array(repeat_padding(seq, size))
    return seq


def repeat_padding(seq, size):
    """
    Parameters
    ----------
    src: list or np.array
    trg: list or np.array
    
    Returns
    -------
    list

    References
    ----------
    1. https://stackoverflow.com/a/60972703
    """
    src = seq
    trg = [0] * size
    data = [src, trg]
    m = len(max(data, key=len))
    r = list(starmap(np.resize, ((e, m) for e in data)))
    return r[0][:size]


def trimmer(seq, size, filler=0):
    """
    Parameters
    ----------
    seq: np.array
        The sequence to be padded.
    size: int
        The size of the output sequence.
    filler: float or int
        Pads with a constant value.
        
    Returns
    -------
    list

    References
    ----------
    1. https://stackoverflow.com/a/30475648
    """
    return list(islice(chain(seq, repeat(filler)), size))
