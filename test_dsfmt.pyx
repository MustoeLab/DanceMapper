
import test

#from dSFMT.dSFMT cimport dsfmt_t, dsfmt_init_gen_rand, dsfmt_genrand_close_open
from dSFMT cimport dsfmt_t, dsfmt_init_gen_rand, dsfmt_genrand_close_open


def test(int nsample):

    cdef dsfmt_t dsfmt
    dsfmt_init_gen_rand(&dsfmt, 0)
    
    sample = []
    
    cdef int i

    for i in xrange(nsample):
        sample.append(dsfmt_genrand_close_open(&dsfmt))
    
    return sample

