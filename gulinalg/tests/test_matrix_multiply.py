"""
Tests on matrix multiply. As the underlying code is now playing with
the TRANSA TRANSB parameters to minimize copying, several tests are
needed to make sure that all cases are handled correctly as its logic
is rather complex.
"""

from __future__ import print_function
from unittest import TestCase, skipIf, skip
import os
import numpy as np
from numpy.testing import assert_allclose
from pkg_resources import parse_version
import gulinalg

M = 75
N = 50
K = 100

# This class tests the cases that code can handle without copy-rearranging of
# any of the input/output arguments.
class TestNoCopy(TestCase):
    # no output specified (operation allocated) ================================
    def test_matrix_multiply_cc(self):
        """matrix multiply two C layout matrices"""
        a = np.ascontiguousarray(np.random.randn(M,N))
        b = np.ascontiguousarray(np.random.randn(N,K))
        res = gulinalg.matrix_multiply(a,b)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_cc_batch(self):
        """matrix multiply two C layout matrices"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N)
            b = np.random.randn(N, K)
            ref = np.dot(a,b)
            a = np.ascontiguousarray(np.stack((a,) * n_outer, axis=0))
            b = np.ascontiguousarray(np.stack((b,) * n_outer, axis=0))
            ref = np.stack((ref,) * n_outer, axis=0)
            res = gulinalg.matrix_multiply(a, b, workers=workers)
            assert_allclose(res, ref)

    def test_matrix_multiply_cf(self):
        """matrix multiply C layout by FORTRAN layout matrices"""
        a = np.ascontiguousarray(np.random.randn(M,N))
        b = np.asfortranarray(np.random.randn(N,K))
        res = gulinalg.matrix_multiply(a,b)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_cf_batch(self):
        """matrix multiply two C layout matrices"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N)
            b = np.random.randn(N, K)
            ref = np.dot(a,b)
            a = np.ascontiguousarray(np.stack((a,) * n_outer, axis=0))
            b = np.asfortranarray(np.stack((b,) * n_outer, axis=0))
            ref = np.stack((ref,) * n_outer, axis=0)
            res = gulinalg.matrix_multiply(a, b, workers=workers)
            assert_allclose(res, ref)

    def test_matrix_multiply_fc(self):
        """matrix multiply FORTRAN layout by C layout matrices"""
        a = np.asfortranarray(np.random.randn(M,N))
        b = np.ascontiguousarray(np.random.randn(N,K))
        res = gulinalg.matrix_multiply(a,b)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_fc_batch(self):
        """matrix multiply two C layout matrices"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N)
            b = np.random.randn(N, K)
            ref = np.dot(a,b)
            a = np.asfortranarray(np.stack((a,) * n_outer, axis=0))
            b = np.ascontiguousarray(np.stack((b,) * n_outer, axis=0))
            ref = np.stack((ref,) * n_outer, axis=0)
            res = gulinalg.matrix_multiply(a, b, workers=workers)
            assert_allclose(res, ref)

    def test_matrix_multiply_ff(self):
        """matrix multiply two FORTRAN layout matrices"""
        a = np.asfortranarray(np.random.randn(M,N))
        b = np.asfortranarray(np.random.randn(N,K))
        res = gulinalg.matrix_multiply(a,b)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_fc_batch(self):
        """matrix multiply two C layout matrices"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N)
            b = np.random.randn(N, K)
            ref = np.dot(a,b)
            a = np.asfortranarray(np.stack((a,) * n_outer, axis=0))
            b = np.asfortranarray(np.stack((b,) * n_outer, axis=0))
            ref = np.stack((ref,) * n_outer, axis=0)
            res = gulinalg.matrix_multiply(a, b, workers=workers)
            assert_allclose(res, ref)

    # C explicit outputs =======================================================
    def test_matrix_multiply_cc_c(self):
        """matrix multiply two C layout matrices, explicit C array output"""
        a = np.ascontiguousarray(np.random.randn(M,N))
        b = np.ascontiguousarray(np.random.randn(N,K))
        res = np.zeros((M,K), order='C')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_cc_c_batch(self):
        """matrix multiply two C layout matrices"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N)
            b = np.random.randn(N, K)
            ref = np.dot(a,b)
            a = np.ascontiguousarray(np.stack((a,) * n_outer, axis=0))
            b = np.ascontiguousarray(np.stack((b,) * n_outer, axis=0))
            ref = np.stack((ref,) * n_outer, axis=0)
            res = np.zeros((n_outer, M, K), order='C')
            gulinalg.matrix_multiply(a, b, workers=workers, out=res)
            assert_allclose(res, ref)

    def test_matrix_multiply_cf_c(self):
        """matrix multiply C layout by FORTRAN layout matrices, explicit C array output"""
        a = np.ascontiguousarray(np.random.randn(M,N))
        b = np.asfortranarray(np.random.randn(N,K))
        res = np.zeros((M,K), order='C')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_cf_c_batch(self):
        """matrix multiply two C layout matrices"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N)
            b = np.random.randn(N, K)
            ref = np.dot(a,b)
            a = np.ascontiguousarray(np.stack((a,) * n_outer, axis=0))
            b = np.asfortranarray(np.stack((b,) * n_outer, axis=0))
            ref = np.stack((ref,) * n_outer, axis=0)
            res = np.zeros((n_outer, M, K), order='C')
            gulinalg.matrix_multiply(a, b, workers=workers, out=res)
            assert_allclose(res, ref)

    def test_matrix_multiply_fc_c(self):
        """matrix multiply FORTRAN layout by C layout matrices, explicit C array output"""
        a = np.asfortranarray(np.random.randn(M,N))
        b = np.ascontiguousarray(np.random.randn(N,K))
        res = np.zeros((M,K), order='C')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_fc_c_batch(self):
        """matrix multiply two C layout matrices"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N)
            b = np.random.randn(N, K)
            ref = np.dot(a,b)
            a = np.asfortranarray(np.stack((a,) * n_outer, axis=0))
            b = np.ascontiguousarray(np.stack((b,) * n_outer, axis=0))
            ref = np.stack((ref,) * n_outer, axis=0)
            res = np.zeros((n_outer, M, K), order='C')
            gulinalg.matrix_multiply(a, b, workers=workers, out=res)
            assert_allclose(res, ref)

    def test_matrix_multiply_ff_c(self):
        """matrix multiply two FORTRAN layout matrices, explicit C array output"""
        a = np.asfortranarray(np.random.randn(M,N))
        b = np.asfortranarray(np.random.randn(N,K))
        res = np.zeros((M,K), order='C')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_ff_c_batch(self):
        """matrix multiply two C layout matrices"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N)
            b = np.random.randn(N, K)
            ref = np.dot(a,b)
            a = np.asfortranarray(np.stack((a,) * n_outer, axis=0))
            b = np.asfortranarray(np.stack((b,) * n_outer, axis=0))
            ref = np.stack((ref,) * n_outer, axis=0)
            res = np.zeros((n_outer, M, K), order='C')
            gulinalg.matrix_multiply(a, b, workers=workers, out=res)
            assert_allclose(res, ref)

    # FORTRAN explicit outputs =================================================
    def test_matrix_multiply_cc_f(self):
        """matrix multiply two C layout matrices, explicit FORTRAN array output"""
        a = np.ascontiguousarray(np.random.randn(M,N))
        b = np.ascontiguousarray(np.random.randn(N,K))
        res = np.zeros((M,K), order='F')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_cc_f_batch(self):
        """matrix multiply two C layout matrices"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N)
            b = np.random.randn(N, K)
            ref = np.dot(a,b)
            a = np.ascontiguousarray(np.stack((a,) * n_outer, axis=0))
            b = np.ascontiguousarray(np.stack((b,) * n_outer, axis=0))
            ref = np.stack((ref,) * n_outer, axis=0)
            res = np.zeros((n_outer, M, K), order='F')
            gulinalg.matrix_multiply(a, b, workers=workers, out=res)
            assert_allclose(res, ref)

    def test_matrix_multiply_cf_f(self):
        """matrix multiply C layout by FORTRAN layout matrices, explicit FORTRAN array output"""
        a = np.ascontiguousarray(np.random.randn(M,N))
        b = np.asfortranarray(np.random.randn(N,K))
        res = np.zeros((M,K), order='F')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_cf_f_batch(self):
        """matrix multiply two C layout matrices"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N)
            b = np.random.randn(N, K)
            ref = np.dot(a,b)
            a = np.ascontiguousarray(np.stack((a,) * n_outer, axis=0))
            b = np.asfortranarray(np.stack((b,) * n_outer, axis=0))
            ref = np.stack((ref,) * n_outer, axis=0)
            res = np.zeros((n_outer, M, K), order='F')
            gulinalg.matrix_multiply(a, b, workers=workers, out=res)
            assert_allclose(res, ref)

    def test_matrix_multiply_fc_f(self):
        """matrix multiply FORTRAN layout by C layout matrices, explicit FORTRAN array output"""
        a = np.asfortranarray(np.random.randn(M,N))
        b = np.ascontiguousarray(np.random.randn(N,K))
        res = np.zeros((M,K), order='F')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_fc_f_batch(self):
        """matrix multiply two C layout matrices"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N)
            b = np.random.randn(N, K)
            ref = np.dot(a,b)
            a = np.asfortranarray(np.stack((a,) * n_outer, axis=0))
            b = np.ascontiguousarray(np.stack((b,) * n_outer, axis=0))
            ref = np.stack((ref,) * n_outer, axis=0)
            res = np.zeros((n_outer, M, K), order='F')
            gulinalg.matrix_multiply(a, b, workers=workers, out=res)
            assert_allclose(res, ref)

    def test_matrix_multiply_ff_f(self):
        """matrix multiply two FORTRAN layout matrices, explicit FORTRAN array output"""
        a = np.asfortranarray(np.random.randn(M,N))
        b = np.asfortranarray(np.random.randn(N,K))
        res = np.zeros((M,K), order='F')
        gulinalg.matrix_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matrix_multiply_ff_f_batch(self):
        """matrix multiply two C layout matrices"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N)
            b = np.random.randn(N, K)
            ref = np.dot(a,b)
            a = np.asfortranarray(np.stack((a,) * n_outer, axis=0))
            b = np.asfortranarray(np.stack((b,) * n_outer, axis=0))
            ref = np.stack((ref,) * n_outer, axis=0)
            res = np.zeros((n_outer, M, K), order='F')
            gulinalg.matrix_multiply(a, b, workers=workers, out=res)
            assert_allclose(res, ref)

# this class test the cases where there is at least one operand/output that
# requires copy/rearranging.
class TestWithCopy(TestCase):
    # No output specified (operation allocated) ================================
    def test_input_non_contiguous_1(self):
        """first input not contiguous"""
        a = np.ascontiguousarray(np.random.randn(M,N,2))[:,:,0]
        b = np.ascontiguousarray(np.random.randn(N,K))
        res = np.zeros((M,K), order='C')
        assert not a.flags.c_contiguous and not a.flags.f_contiguous
        gulinalg.matrix_multiply(a, b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_input_non_contiguous_1_batch(self):
        """first input not contiguous"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N, 2)
            b = np.random.randn(N, K)
            ref = np.dot(a[..., 0], b)
            a = np.ascontiguousarray(np.stack((a,) * n_outer, axis=0))[..., 0]
            b = np.ascontiguousarray(np.stack((b,) * n_outer, axis=0))
            assert not a.flags.c_contiguous and not a.flags.f_contiguous
            ref = np.stack((ref,) * n_outer, axis=0)
            res = np.zeros((n_outer, M, K), order='C')
            gulinalg.matrix_multiply(a, b, workers=workers, out=res)
            assert_allclose(res, ref)

    def test_input_non_contiguous_2(self):
        """second input not contiguous"""
        a = np.ascontiguousarray(np.random.randn(M,N))
        b = np.ascontiguousarray(np.random.randn(N,K,2))[:,:,0]
        res = np.zeros((M,K), order='C')
        assert not b.flags.c_contiguous and not b.flags.f_contiguous
        gulinalg.matrix_multiply(a, b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_input_non_contiguous_2_batch(self):
        """second input not contiguous"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N)
            b = np.random.randn(N, K, 2)
            ref = np.dot(a, b[..., 0])
            a = np.ascontiguousarray(np.stack((a,) * n_outer, axis=0))
            b = np.ascontiguousarray(np.stack((b,) * n_outer, axis=0))[..., 0]
            assert not b.flags.c_contiguous and not b.flags.f_contiguous
            ref = np.stack((ref,) * n_outer, axis=0)
            res = np.zeros((n_outer, M, K), order='C')
            gulinalg.matrix_multiply(a, b, workers=workers, out=res)
            assert_allclose(res, ref)

    def test_input_non_contiguous_3(self):
        """neither input contiguous"""
        a = np.ascontiguousarray(np.random.randn(M,N,2))[:,:,0]
        b = np.ascontiguousarray(np.random.randn(N,K,2))[:,:,0]
        res = np.zeros((M,K), order='C')
        assert not a.flags.c_contiguous and not a.flags.f_contiguous
        assert not b.flags.c_contiguous and not b.flags.f_contiguous
        gulinalg.matrix_multiply(a, b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_input_non_contiguous_3_batch(self):
        """neither input contiguous"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N, 2)
            b = np.random.randn(N, K, 2)
            ref = np.dot(a[..., 0], b[..., 0])
            a = np.ascontiguousarray(np.stack((a,) * n_outer, axis=0))[..., 0]
            b = np.ascontiguousarray(np.stack((b,) * n_outer, axis=0))[..., 0]
            assert not a.flags.c_contiguous and not a.flags.f_contiguous
            assert not b.flags.c_contiguous and not b.flags.f_contiguous
            ref = np.stack((ref,) * n_outer, axis=0)
            res = np.zeros((n_outer, M, K), order='C')
            gulinalg.matrix_multiply(a, b, workers=workers, out=res)
            assert_allclose(res, ref)

    def test_output_non_contiguous(self):
        """output not contiguous"""
        a = np.ascontiguousarray(np.random.randn(M,N))
        b = np.ascontiguousarray(np.random.randn(N,K))
        res = np.zeros((M,K,2), order='C')[:,:,0]
        assert not res.flags.c_contiguous and not res.flags.f_contiguous
        gulinalg.matrix_multiply(a, b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_output_non_contiguous_batch(self):
        """output not contiguous"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N)
            b = np.random.randn(N, K)
            ref = np.dot(a, b)
            a = np.ascontiguousarray(np.stack((a,) * n_outer, axis=0))
            b = np.ascontiguousarray(np.stack((b,) * n_outer, axis=0))
            ref = np.stack((ref,) * n_outer, axis=0)
            res = np.zeros((n_outer, M, K, 2), order='C')[..., 0]
            assert not res.flags.c_contiguous and not res.flags.f_contiguous
            gulinalg.matrix_multiply(a, b, workers=workers, out=res)
            assert_allclose(res, ref)

    def test_all_non_contiguous(self):
        """neither input nor output contiguous"""
        a = np.ascontiguousarray(np.random.randn(M,N,2))[:,:,0]
        b = np.ascontiguousarray(np.random.randn(N,K,2))[:,:,0]
        res = np.zeros((M,K,2), order='C')[:,:,0]
        assert not a.flags.c_contiguous and not a.flags.f_contiguous
        assert not b.flags.c_contiguous and not b.flags.f_contiguous
        assert not res.flags.c_contiguous and not res.flags.f_contiguous
        gulinalg.matrix_multiply(a, b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)


    def test_all_non_contiguous_batch(self):
        """neither input nor output contiguous"""
        n_outer = 4
        for workers in [1, -1]:
            a = np.random.randn(M, N, 2)
            b = np.random.randn(N, K, 2)
            ref = np.dot(a[..., 0], b[..., 0])
            a = np.ascontiguousarray(np.stack((a,) * n_outer, axis=0))[..., 0]
            b = np.ascontiguousarray(np.stack((b,) * n_outer, axis=0))[..., 0]
            ref = np.stack((ref,) * n_outer, axis=0)
            res = np.zeros((n_outer, M, K, 2), order='C')[..., 0]
            assert not a.flags.c_contiguous and not a.flags.f_contiguous
            assert not b.flags.c_contiguous and not b.flags.f_contiguous
            assert not res.flags.c_contiguous and not res.flags.f_contiguous
            gulinalg.matrix_multiply(a, b, workers=workers, out=res)
            assert_allclose(res, ref)

    def test_stride_tricks(self):
        """test that matrices that are contiguous but have their dimension
        overlapped *copy*, as BLAS does not support them"""
        a = np.ascontiguousarray(np.random.randn(M + N))
        a = np.lib.stride_tricks.as_strided(a, shape=(M,N),
                                            strides=(a.itemsize, a.itemsize))
        b = np.ascontiguousarray(np.random.randn(N,K))
        res = gulinalg.matrix_multiply(a,b)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    @skipIf(os.environ.get('GULINALG_I32_TEST') is None,
                           reason="needs 32gb; define the env var to run.")
    def test_i32(self):
        """Test matrix sizes > int32. Requires > 32Gb memory."""
        a = np.ones((8, 2**28))
        b = np.ones((2**28, 8))
        c = gulinalg.matrix_multiply(a, b)
        assert_allclose(c, np.ones((8, 8)) * 2**28)


# Some simple tests showing that the gufunc stuff works
class TestVector(TestCase):
    def test_vector(self):
        """test vectorized matrix multiply"""
        a = np.ascontiguousarray(np.random.randn(10, M, N))
        b = np.ascontiguousarray(np.random.randn(10, N, K))

        res = gulinalg.matrix_multiply(a,b)
        assert res.shape == (10, M, K)
        ref = np.stack([np.dot(a[i], b[i]) for i in range(len(a))])
        assert_allclose(res, ref)


    def test_broadcast(self):
        """test broadcast matrix multiply"""
        a = np.ascontiguousarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(10, N, K))

        res = gulinalg.matrix_multiply(a,b)
        assert res.shape == (10, M, K)
        ref = np.stack([np.dot(a, b[i]) for i in range(len(b))])
        assert_allclose(res, ref)

    def test_vectorized_matvec(self):
        """test vectorized matrix multiply"""
        a = np.ascontiguousarray(np.random.randn(10, M, N))
        b = np.ascontiguousarray(np.random.randn(10, N))

        res = gulinalg.matvec_multiply(a,b)
        assert res.shape == (10, M)
        ref = np.stack([np.dot(a[i], b[i]) for i in range(len(a))])
        assert_allclose(res, ref)


    def test_broadcast_matvec(self):
        """test broadcast matrix multiply"""
        a = np.ascontiguousarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(10, N))

        res = gulinalg.matvec_multiply(a,b)
        assert res.shape == (10, M)
        ref = np.stack([np.dot(a, b[i]) for i in range(len(b))])
        assert_allclose(res, ref)

class TestMatvecNoCopy(TestCase):
    def test_matvec_multiply_cf_c(self):
        """matrix multiply C layout by FORTRAN layout vector, explicit C array output"""
        a = np.ascontiguousarray(np.random.randint(100, size=(M,N)))
        b = np.asfortranarray(np.random.randint(N, size=(N)))
        res = np.zeros(M, order='C')
        gulinalg.matvec_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

    def test_matvec_multiply_fc_f(self):
        """matrix multiply FORTRAN layout by C layout vector, explicit FORTRAN array output"""
        a = np.asfortranarray(np.random.randint(100, size=(M,N)))
        b = np.ascontiguousarray(np.random.randint(N, size=(N)))
        res = np.zeros(M, order='F')
        gulinalg.matvec_multiply(a,b, out=res)
        ref = np.dot(a,b)
        assert_allclose(res, ref)

