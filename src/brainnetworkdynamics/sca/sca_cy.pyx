import numpy as np
cimport numpy as np
from libc.math cimport sqrt, log, exp


cdef float _pearson_correlation(float[:] x, float[:] y):
    cdef int i, n = x.shape[0]
    cdef float x_mean = 0.0, y_mean = 0.0
    cdef float xy_sum = 0.0, xx_sum = 0.0, yy_sum = 0.0
    
    for i in range(n):
        x_mean += x[i]
        y_mean += y[i]
    x_mean /= n
    y_mean /= n

    for i in range(n):
        xy_sum += (x[i] - x_mean)*(y[i] - y_mean)
        xx_sum += (x[i] - x_mean)*(x[i] - x_mean)
        yy_sum += (y[i] - y_mean)*(y[i] - y_mean)

    xy_cov = xy_sum/(n - 1)
    x_std = sqrt(xx_sum/(n - 1))
    y_std = sqrt(yy_sum/(n - 1))

    return xy_cov/x_std/y_std


cdef float _fisher_transform(float r):
    return 0.5*log((1 + r/1.12)/(1 - r/1.12))


cdef float _inverse_fisher_transform(float z):
    return 1.12*(exp(2*z) - 1)/(exp(2*z) + 1)


cdef float _scaled_correlation(
    float[:] x,
    float[:] y,
    int s,
    bint fisher_transform
):
    cdef int n_x, n_y
    cdef int T, K, i

    n_x = x.shape[0]
    n_y = y.shape[0]

    T = n_x
    K = T//s

    if fisher_transform == 0:
        r_s = 0.0
        for i in range(K):
            r_s += _pearson_correlation(
                x[i*s:(i + 1)*s],
                y[i*s:(i + 1)*s]
            )
        r_s /= K
    else:
        z = 0.0
        for i in range(K):
            z += _fisher_transform(
                _pearson_correlation(
                    x[i*s:(i + 1)*s],
                    y[i*s:(i + 1)*s]
                )
            )
        z /= K
        r_s = _inverse_fisher_transform(z)
    
    return r_s


cpdef cross_correlation(
    float[:] corr_coeff_arr,
    float[:] x,
    float[:] y,
    int max_shift_size,
    int scale_size,
    bint fisher_transform
):
    cdef int n_x, n_y
    cdef int shift
    cdef int start, end                       # <-- MODIFIED

    n_x = x.shape[0]
    n_y = y.shape[0]

    # Common interval used for all lags
    start = max_shift_size                          # <-- MODIFIED
    end = min(n_x, n_y) - max_shift_size            # <-- MODIFIED

    if scale_size > 0:

        # SCA
        for shift in range(-max_shift_size, max_shift_size + 1):  # <-- MODIFIED
            corr_coeff_arr[shift + max_shift_size] = _scaled_correlation(
                x[start:end],                         # <-- MODIFIED
                y[start + shift:end + shift],         # <-- MODIFIED
                scale_size,
                fisher_transform
            )

    else:

        # CC
        for shift in range(-max_shift_size, max_shift_size + 1):  # <-- MODIFIED
            corr_coeff_arr[shift + max_shift_size] = _pearson_correlation(
                x[start:end],                         # <-- MODIFIED
                y[start + shift:end + shift]          # <-- MODIFIED
            )
