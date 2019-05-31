
import numpy as np

try:
    import test_dsfmt

    comparison = np.array(test_dsfmt.test(1000))

    reference = []
    with open('dSFMT/dSFMT.19937.out.txt') as inp:
    
        inp.readline()
        inp.readline()
    
        for i in range(250):
            reference.extend(map(float, inp.readline().split()))

    
    reference = np.array(reference)-1

    if np.allclose(comparison, reference, atol=1e-15):
        print('dSFMT test PASSED')
    else:
        exit('WARNING: dSFMT test FAILED!!!!')


except ImportError:
    exit('test_dsfmt not compiled correctly')

except:
    exit('WARNING: dSFMT test FAILED!!!!')




if 0:
    import matplotlib.pyplot as plot
    sample = test.test(5000000)
    plot.hist(sample, bins=100)
    plot.show()

