import numpy as np


def sample_triple(a):
    '''
    sample a triplet out of a list
    '''
    samples = [i.item() for i in np.random.choice(a, 3, False)]
    return samples
