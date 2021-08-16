import numpy as np
from libc.math cimport isnan

cimport numpy as cnp
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True) 
cpdef compute_period_averages(cython.floating[:] array, int period):
    cdef cython.floating[:] acum_sum = np.zeros(period)
    cdef cython.floating[:] acum_count = np.zeros(period)
    cdef int k, counter_pos
    cdef cython.floating x_k
    
    for k in range(len(array)):
        counter_pos = k % period
        
        x_k = array[k]
        if not isnan(x_k):
            acum_sum[counter_pos] += x_k
            acum_count[counter_pos] += 1

    for k in range(period):
        acum_sum[k] /= acum_count[k]
        
    return np.array(acum_sum)


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef compute_ses(cython.floating[:] y, cython.floating alpha):
    cdef int n_obs = len(y)  
    cdef int t
    cdef cython.floating[:] fh = np.empty(n_obs + 1)
    cdef cython.floating alpha_y, alpha_fh

    fh[0] = y[0]
    fh[1] = y[0]
    
    for t in range(2, n_obs + 1):
        fh[t] = alpha * y[t - 1] + (1 - alpha) * fh[t - 1]
        
    return (fh[:n_obs], fh[n_obs])

