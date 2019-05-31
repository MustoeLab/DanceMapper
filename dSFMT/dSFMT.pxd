

cdef extern from "dSFMT.h":
    
    cdef struct DSFMT_T:
        pass
    
    ctypedef DSFMT_T dsfmt_t 

    ctypedef unsigned int uint32_t

    void dsfmt_init_gen_rand(dsfmt_t*, uint32_t)
    
    double dsfmt_genrand_close_open(dsfmt_t*)
    

