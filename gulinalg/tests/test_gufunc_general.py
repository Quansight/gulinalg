"""
Tests BLAS functions. Since it supports C as well as Fortran
matrix, that leads to various combinations of matrices to test.
"""
from itertools import product
from functools import partial
from unittest import TestCase, skipIf
import numpy as np
from numpy.testing import (assert_allclose, assert_,
                           assert_raises)
from pkg_resources import parse_version
import gulinalg

M = 75
N = 50
K = 100
n_batch = 8


class TestInner1d(TestCase):

    def test_real(self):
        a = np.random.randn(N)
        b = np.random.randn(N)
        res = gulinalg.inner1d(a, b)
        ref = np.sum(a * b)
        assert_allclose(res, ref)

    @skipIf(True, reason="returns garbage")
    def test_complex(self):
        a = np.random.randn(N) + 1j * np.random.randn(N)
        b = np.random.randn(N) + 1j * np.random.randn(N)
        res = gulinalg.inner1d(a, b)
        ref = np.sum(a * b)
        assert_allclose(res, ref)

    def test_real_vector(self):
        a = np.random.randn(n_batch, N)
        b = np.random.randn(n_batch, N)
        for workers in [1, -1]:
            res = gulinalg.inner1d(a, b, workers=workers)
            ref = np.sum(a * b, axis=-1)
            assert_allclose(res, ref)

    @skipIf(True, "segfaults")
    def test_complex_vector(self):
        a = np.random.randn(n_batch, N) + 1j * np.random.randn(n_batch, N)
        b = np.random.randn(n_batch, N) + 1j * np.random.randn(n_batch, N)
        for workers in [1, -1]:
            res = gulinalg.inner1d(a, b, workers=workers)
            ref = np.sum(a * b, axis=-1)
            assert_allclose(res, ref)


class TestDotc1d(TestCase):

    @skipIf(True, reason="returns garbage")
    def test_complex(self):
        a = np.random.randn(N) + 1j * np.random.randn(N)
        b = np.random.randn(N) + 1j * np.random.randn(N)
        res = gulinalg.dotc1d(a, b)
        ref = np.sum(np.conj(a) * b)
        assert_allclose(res, ref)

    @skipIf(True, "segfaults")
    def test_complex_vector(self):
        a = np.random.randn(n_batch, N) + 1j * np.random.randn(n_batch, N)
        b = np.random.randn(n_batch, N) + 1j * np.random.randn(n_batch, N)
        for workers in [1, -1]:
            res = gulinalg.dotc1d(a, b, workers=workers)
            ref = np.sum(np.conj(a) * b, axis=-1)
            assert_allclose(res, ref)


class TestInnerwt(TestCase):

    def test_real(self):
        a = np.random.randn(N)
        b = np.random.randn(N)
        c = np.random.randn(N)
        res = gulinalg.innerwt(a, b, c)
        ref = np.sum(a * b * c)
        assert_allclose(res, ref)

    def test_complex(self):
        a = np.random.randn(N) + 1j * np.random.randn(N)
        b = np.random.randn(N) + 1j * np.random.randn(N)
        c = np.random.randn(N) + 1j * np.random.randn(N)
        res = gulinalg.innerwt(a, b, c)
        ref = np.sum(a * b * c)
        assert_allclose(res, ref)

    def test_real_vector(self):
        a = np.random.randn(n_batch, N)
        b = np.random.randn(n_batch, N)
        c = np.random.randn(n_batch, N)
        for workers in [1, -1]:
            res = gulinalg.innerwt(a, b, c, workers=workers)
            ref = np.sum(a * b * c, axis=-1)
            assert_allclose(res, ref)

    def test_complex_vector(self):
        a = np.random.randn(n_batch, N) + 1j * np.random.randn(n_batch, N)
        b = np.random.randn(n_batch, N) + 1j * np.random.randn(n_batch, N)
        c = np.random.randn(n_batch, N) + 1j * np.random.randn(n_batch, N)
        for workers in [1, -1]:
            res = gulinalg.innerwt(a, b, c, workers=workers)
            ref = np.sum(a * b * c, axis=-1)
            assert_allclose(res, ref)


class TestMatvecMultiplyNoCopy(TestCase):
    """
    Tests the cases that code can handle without copy-rearranging of any of
    the input/output arguments.
    """
    def test_matvec_multiply_c(self):
        """Multiply C layout matrix with vector"""
        a = np.ascontiguousarray(np.random.randn(M, N))
        b = np.random.randn(N)
        res = gulinalg.matvec_multiply(a, b)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_multiply_f(self):
        """Multiply FORTRAN layout matrix with vector"""
        a = np.asfortranarray(np.random.randn(M, N))
        b = np.random.randn(N)
        res = gulinalg.matvec_multiply(a, b)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_multiply_cv_c(self):
        """Test for explicit C array output for C layout input matrix"""
        a = np.ascontiguousarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(N))
        res = np.zeros(M, order='C')
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_multiply_fv_c(self):
        """Test for explicit C array output for FORTRAN layout input matrix"""
        a = np.asfortranarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(N))
        res = np.zeros(M, order='C')
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_multiply_cv_f(self):
        """Test for explicit FORTRAN array output for C layout input matrix"""
        a = np.ascontiguousarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(N))
        res = np.zeros(M, order='F')
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_multiply_fv_f(self):
        """Test for explicit FORTRAN array output for F layout input matrix"""
        a = np.asfortranarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(N))
        res = np.zeros(M, order='F')
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_multiply_for_complex_numbers(self):
        """Test for complex numbers input."""
        a = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + -8j]])
        b = np.array([1 - 2j, 4 + 5j])
        res = gulinalg.matvec_multiply(a, b)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_matvec_size_zero_matrix(self):
        """Test matrix of size zero"""
        a = np.random.randn(0, 2)
        b = np.random.randn(2)
        res = gulinalg.matvec_multiply(a, b)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_matvec_size_zero_vector(self):
        """Test vector of size zero"""
        a = np.random.randn(2, 0)
        b = np.random.randn(0)
        res = gulinalg.matvec_multiply(a, b)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_size_one_vector(self):
        """Test vector of size one"""
        a = np.random.randn(1, 1)
        b = np.random.randn(1)
        res = gulinalg.matvec_multiply(a, b)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_multiply_batch_b(self):
        """Multiply C layout matrix with stack of vectors"""
        a = np.ascontiguousarray(np.random.randn(M, N))
        b = np.random.randn(n_batch, N)
        for workers in [1, -1]:
            res = gulinalg.matvec_multiply(a, b)
            ref = np.matmul(a, b[:, :, np.newaxis])[..., 0]
            assert_allclose(res, ref)

    def test_matvec_multiply_batch_a(self):
        """Multiply C layout stack of matrices with a vector"""
        a = np.ascontiguousarray(np.random.randn(n_batch, M, N))
        b = np.random.randn(N)
        for workers in [1, -1]:
            res = gulinalg.matvec_multiply(a, b)
            ref = np.matmul(a, b[:, np.newaxis])[..., 0]
            assert_allclose(res, ref)

    def test_matvec_multiply_batch_both(self):
        """Multiply C layout stack of matrices and vectors"""
        a = np.ascontiguousarray(np.random.randn(n_batch, M, N))
        b = np.random.randn(n_batch, N)
        for workers in [1, -1]:
            res = gulinalg.matvec_multiply(a, b)
            ref = np.matmul(a, b[:, :, np.newaxis])[..., 0]
            assert_allclose(res, ref)

    def test_matvec_multiply_batch_both_out(self):
        """Multiply C layout stack of matrices and vectors"""
        a = np.ascontiguousarray(np.random.randn(n_batch, M, N))
        b = np.random.randn(n_batch, N)
        res = np.zeros((n_batch, M), dtype=a.dtype)
        for workers in [1, -1]:
            gulinalg.matvec_multiply(a, b, out=res)
            ref = np.matmul(a, b[:, :, np.newaxis])[..., 0]
            assert_allclose(res, ref)


class TestMatvecMultiplyWithCopy(TestCase):
    """
    Test the cases where there is at least one operand/output that requires
    copy/rearranging.
    """
    def test_input_non_contiguous_1(self):
        """First input not contiguous"""
        a = np.ascontiguousarray(np.random.randn(M, N, 2))[:, :, 0]
        b = np.ascontiguousarray(np.random.randn(N))
        res = np.zeros(M, order='C')
        assert not a.flags.c_contiguous and not a.flags.f_contiguous
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_input_non_contiguous_2(self):
        """Second input not contiguous"""
        a = np.ascontiguousarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(N, 2))[:, 0]
        res = np.zeros(M, order='C')
        assert not b.flags.c_contiguous and not b.flags.f_contiguous
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_input_non_contiguous_3(self):
        """Neither input contiguous"""
        a = np.ascontiguousarray(np.random.randn(M, N, 2))[:, :, 0]
        b = np.ascontiguousarray(np.random.randn(N, 2))[:, 0]
        res = np.zeros(M, order='C')
        assert not a.flags.c_contiguous and not a.flags.f_contiguous
        assert not b.flags.c_contiguous and not b.flags.f_contiguous
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_output_non_contiguous(self):
        """Output not contiguous"""
        a = np.ascontiguousarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(N))
        res = np.zeros((M, 2), order='C')[:, 0]
        assert not res.flags.c_contiguous and not res.flags.f_contiguous
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_all_non_contiguous(self):
        """Neither input nor output contiguous"""
        a = np.ascontiguousarray(np.random.randn(M, N, 2))[:, :, 0]
        b = np.ascontiguousarray(np.random.randn(N, 2))[:, 0]
        res = np.zeros((M, 2), order='C')[:, 0]
        assert not a.flags.c_contiguous and not a.flags.f_contiguous
        assert not b.flags.c_contiguous and not b.flags.f_contiguous
        assert not res.flags.c_contiguous and not res.flags.f_contiguous
        gulinalg.matvec_multiply(a, b, out=res)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_stride_tricks(self):
        """Test that matrices that are contiguous but have their dimension
        overlapped *copy*, as BLAS does not support them"""
        a = np.ascontiguousarray(np.random.randn(M + N))
        a = np.lib.stride_tricks.as_strided(a,
                                            shape=(M, N),
                                            strides=(a.itemsize, a.itemsize))
        b = np.ascontiguousarray(np.random.randn(N))
        res = gulinalg.matvec_multiply(a, b)
        ref = np.dot(a, b)
        assert_allclose(res, ref)

    def test_matvec_multiply_batch_both_out_non_contiguous(self):
        """Multiply C layout stack of matrices and vectors"""
        a = np.random.randn(n_batch, M, N, 2)[..., 0]
        b = np.random.randn(n_batch, N, 2)[..., 0]
        res = np.zeros((n_batch, M, 2), dtype=a.dtype)[..., 0]
        for workers in [1, -1]:
            gulinalg.matvec_multiply(a, b, out=res)
            ref = np.matmul(a, b[:, :, np.newaxis])[..., 0]
            assert_allclose(res, ref)


class TestMatvecMultiplyVector(TestCase):
    """Tests showing that the gufunc stuff works"""
    def test_vector(self):
        """test vectorized matrix multiply"""
        a = np.ascontiguousarray(np.random.randn(10, M, N))
        b = np.ascontiguousarray(np.random.randn(10, N))
        res = gulinalg.matvec_multiply(a, b)
        assert res.shape == (10, M)
        ref = np.stack([np.dot(a[i], b[i]) for i in range(len(a))])
        assert_allclose(res, ref)

    def test_broadcast(self):
        """test broadcast matrix multiply"""
        a = np.ascontiguousarray(np.random.randn(M, N))
        b = np.ascontiguousarray(np.random.randn(10, N))
        res = gulinalg.matvec_multiply(a, b)
        assert res.shape == (10, M)
        ref = np.stack([np.dot(a, b[i]) for i in range(len(b))])
        assert_allclose(res, ref)

    def test_nan_handling(self):
        """NaN in one output shouldn't contaminate remaining outputs"""
        a = np.eye(2)
        b = np.array([[1.0, 2.0], [np.nan, 1.0]])
        ref = np.array([[1., 2.], [np.nan, np.nan]])
        res = gulinalg.matvec_multiply(a, b)
        assert_allclose(res, ref)

    def test_infinity_handling(self):
        """Infinity in one output shouldn't contaminate remaining outputs"""
        a = np.eye(2)
        b = np.array([[1.0, 2.0], [np.inf, 1.0]])
        ref = np.array([[1., 2.], [np.inf, np.nan]])
        res = gulinalg.matvec_multiply(a, b)
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_size_zero_vector(self):
        """Test broadcasting for vector of size zero"""
        a = np.ascontiguousarray(np.random.randn(10, 2, 0))
        b = np.ascontiguousarray(np.random.randn(10, 0))
        res = gulinalg.matvec_multiply(a, b)
        assert res.shape == (10, 2)
        ref = np.stack([np.dot(a[i], b[i]) for i in range(len(a))])
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_size_zero_matrix(self):
        """Test broadcasting for matrix of size zero"""
        a = np.ascontiguousarray(np.random.randn(10, 0, 2))
        b = np.ascontiguousarray(np.random.randn(10, 2))
        res = gulinalg.matvec_multiply(a, b)
        assert res.shape == (10, 0)
        ref = np.stack([np.dot(a[i], b[i]) for i in range(len(a))])
        assert_allclose(res, ref)

    def test_size_one_vector(self):
        """Test broadcasting for vector of size one"""
        a = np.ascontiguousarray(np.random.randn(10, 1, 1))
        b = np.ascontiguousarray(np.random.randn(10, 1))
        res = gulinalg.matvec_multiply(a, b)
        assert res.shape == (10, 1)
        ref = np.stack([np.dot(a[i], b[i]) for i in range(len(a))])
        assert_allclose(res, ref)


class TestUpdateRank1Copy(TestCase):
    """
    Tests the cases that code can handle without copy-rearranging of any of
    the input/output arguments.
    """
    def test_update_rank1_c(self):
        """Rank update on C layout matrix"""
        a = np.random.randn(M)
        b = np.random.randn(N)
        c = np.ascontiguousarray(np.random.randn(M, N))
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(a.reshape(M, 1), b.reshape(1, N)) + c
        assert_allclose(res, ref)

    def test_update_rank1_f(self):
        """Rank update on F layout matrix"""
        a = np.random.randn(M)
        b = np.random.randn(N)
        c = np.asfortranarray(np.random.randn(M, N))
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(a.reshape(M, 1), b.reshape(1, N)) + c
        assert_allclose(res, ref)

    def test_update_rank1_for_complex_numbers(self):
        """Test for complex numbers"""
        a = np.array([1 + 3j, 3 - 4j])
        b = np.array([1 - 2j, 4 + 5j])
        c = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + -8j]])
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(a.reshape(2, 1), b.conj().reshape(1, 2)) + c
        assert_allclose(res, ref)

    def test_update_rank1_for_complex_numbers_no_conjugate_transpose(self):
        """Test for complex numbers but no conjuage transpose"""
        a = np.array([1 + 3j, 3 - 4j])
        b = np.array([1 - 2j, 4 + 5j])
        c = np.array([[1 + 2j, 3 + 4j], [5 + 6j, 7 + -8j]])
        res = gulinalg.update_rank1(a, b, c, conjugate=False)
        ref = np.dot(a.reshape(2, 1), b.reshape(1, 2)) + c
        assert_allclose(res, ref)

    def test_update_rank1_c_c(self):
        """Rank1 update on C layout matrix, explicit C array output"""
        a = np.array([2, 3, 4])
        b = np.array([1, 3, 4, 5])
        c = np.arange(1, 13).reshape(3, 4)
        res = np.zeros((3, 4), order='C')
        gulinalg.update_rank1(a, b, c, out=res)
        ref = np.dot(a.reshape(3, 1), b.reshape(1, 4)) + c
        assert_allclose(res, ref)

    def test_update_rank1_f_c(self):
        """Rank1 update on F layout matrix, explicit C array output"""
        a = np.array([2, 3, 4])
        b = np.array([1, 3, 4, 5])
        c = np.asfortranarray(np.arange(1, 13).reshape(3, 4))
        res = np.zeros((3, 4), order='C')
        gulinalg.update_rank1(a, b, c, out=res)
        ref = np.dot(a.reshape(3, 1), b.reshape(1, 4)) + c
        assert_allclose(res, ref)

    def test_update_rank1_c_f(self):
        """Rank1 update on C layout matrix, explicit F array output"""
        a = np.array([2, 3, 4])
        b = np.array([1, 3, 4, 5])
        c = np.arange(1, 13).reshape(3, 4)
        res = np.zeros((3, 4), order='F')
        gulinalg.update_rank1(a, b, c, out=res)
        ref = np.dot(a.reshape(3, 1), b.reshape(1, 4)) + c
        assert_allclose(res, ref)

    def test_update_rank1_f_f(self):
        """Rank1 update on F layout matrix, explicit F array output"""
        a = np.array([2, 3, 4])
        b = np.array([1, 3, 4, 5])
        c = np.asfortranarray(np.arange(1, 13).reshape(3, 4))
        res = np.zeros((3, 4), order='F')
        gulinalg.update_rank1(a, b, c, out=res)
        ref = np.dot(a.reshape(3, 1), b.reshape(1, 4)) + c
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_size_zero_vector(self):
        """Test vector input of size zero"""
        a = np.zeros(1)
        b = np.zeros(0)
        c = np.ascontiguousarray(np.random.randn(1, 0))
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(np.zeros((1, 0)), np.zeros((0, 0))) + c
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_size_zero_matrix(self):
        """Test matrix input of size zero"""
        a = np.zeros(0)
        b = np.zeros(2)
        c = np.full((0, 2), np.nan)
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(np.zeros((0, 0)), np.zeros((0, 2))) + c
        assert_allclose(res, ref)

    def test_size_one_vector(self):
        """Test vector inputs of size one"""
        a = np.random.randn(1)
        b = np.random.randn(1)
        c = np.ascontiguousarray(np.random.randn(1, 1))
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(a.reshape(1, 1), b.reshape(1, 1)) + c
        assert_allclose(res, ref)


class TestUpdateRank1WithCopy(TestCase):
    """
    Test the cases where there is at least one operand/output that requires
    copy/rearranging.
    """
    def test_input_non_contiguous_vectors(self):
        """Not contiguous vector inputs"""
        a = np.ascontiguousarray(np.random.randn(M, N, 2))[:, 0, 0]
        b = np.ascontiguousarray(np.random.randn(M, N, 2))[0, :, 0]
        c = np.ascontiguousarray(np.random.randn(M, N))
        assert not a.flags.c_contiguous and not a.flags.f_contiguous
        assert not b.flags.c_contiguous and not b.flags.f_contiguous
        for workers in [1, -1]:
            res = gulinalg.update_rank1(a, b, c, workers=workers)
            ref = np.dot(a.reshape(M, 1), b.reshape(1, N)) + c
            assert_allclose(res, ref)

    def test_input_non_contiguous_matrix(self):
        """Non contiguous matrix input"""
        a = np.random.randn(M)
        b = np.random.randn(N)
        c = np.ascontiguousarray(np.random.randn(M, N, 2))[:, :, 0]
        assert not c.flags.c_contiguous and not c.flags.f_contiguous
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(a.reshape(M, 1), b.reshape(1, N)) + c
        assert_allclose(res, ref)

    def test_output_non_contiguous(self):
        """Output not contiguous"""
        a = np.random.randn(M)
        b = np.random.randn(N)
        c = np.ascontiguousarray(np.random.randn(M, N))
        res = np.zeros((M, N, 2), order='C')[:, :, 0]
        gulinalg.update_rank1(a, b, c, out=res)
        ref = np.dot(a.reshape(M, 1), b.reshape(1, N)) + c
        assert_allclose(res, ref)

    def test_stride_tricks(self):
        """test that matrices that are contiguous but have their dimension
        overlapped *copy*, as BLAS does not support them"""
        a = np.random.randn(M)
        b = np.random.randn(N)
        c = np.ascontiguousarray(np.random.randn(M + N))
        c = np.lib.stride_tricks.as_strided(a,
                                            shape=(M, N),
                                            strides=(c.itemsize, c.itemsize))
        res = gulinalg.update_rank1(a, b, c)
        ref = np.dot(a.reshape(M, 1), b.reshape(1, N)) + c
        assert_allclose(res, ref)


class TestUpdateRank1Vector(TestCase):
    """Tests showing that the gufunc stuff works"""
    def test_vector(self):
        """test vectorized rank1 update"""
        a = np.ascontiguousarray(np.random.randn(10, M))
        b = np.ascontiguousarray(np.random.randn(10, N))
        c = np.ascontiguousarray(np.random.randn(10, M, N))
        for workers in [1, -1]:
            res = gulinalg.update_rank1(a, b, c, workers=workers)
            assert res.shape == (10, M, N)
            ref = np.stack([np.dot(a[i].reshape(M, 1), b[i].reshape(1, N)) + c[i]
                            for i in range(len(c))])
            assert_allclose(res, ref)

    def test_broadcast(self):
        """test broadcast rank1 update"""
        a = np.ascontiguousarray(np.random.randn(10, M))
        b = np.ascontiguousarray(np.random.randn(10, N))
        c = np.ascontiguousarray(np.random.randn(M, N))
        for workers in [1, -1]:
            res = gulinalg.update_rank1(a, b, c, workers=workers)
            assert res.shape == (10, M, N)
            ref = np.stack([np.dot(a[i].reshape(M, 1), b[i].reshape(1, N)) + c
                            for i in range(len(b))])
            assert_allclose(res, ref)

    def test_nan_handling(self):
        """NaN in one output shouldn't contaminate remaining outputs"""
        a = np.array([[1, 2], [1, np.nan]])
        b = np.array([3, 4])
        c = np.array([[1, 2], [3, 4]])
        ref = np.array([[[4, 6], [9, 12]],
                        [[4, 6], [np.nan, np.nan]]])
        res = gulinalg.update_rank1(a, b, c)
        assert_allclose(res, ref)

    def test_infinity_handling(self):
        """Infinity in one output shouldn't contaminate remaining outputs"""
        a = np.array([[1, 2], [1, np.inf]])
        b = np.array([3, 4])
        c = np.array([[1, 2], [3, 4]])
        ref = np.array([[[4, 6], [9, 12]],
                        [[4, 6], [np.inf, np.inf]]])
        res = gulinalg.update_rank1(a, b, c)
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_size_zero_vector(self):
        """Test broadcasting for matrix input of size zero"""
        a = np.ascontiguousarray(np.random.randn(10, 1))
        b = np.ascontiguousarray(np.random.randn(10, 0))
        c = np.ascontiguousarray(np.random.randn(10, 1, 0))
        res = gulinalg.update_rank1(a, b, c)
        assert res.shape == (10, 1, 0)
        ref = np.stack([np.dot(np.zeros((1, 0)), np.zeros((0, 0))) + c[i]
                        for i in range(len(c))])
        assert_allclose(res, ref)

    @skipIf(parse_version(np.__version__) < parse_version('1.13'),
            "Prior to 1.13, numpy low level iterators didn't support removing "
            "empty axis. So gufunc couldn't be called with empty inner loop")
    def test_size_zero_matrix(self):
        """Test broadcasting for matrix input of size zero"""
        a = np.ascontiguousarray(np.random.randn(10, 0))
        b = np.ascontiguousarray(np.random.randn(10, 2))
        c = np.ascontiguousarray(np.random.randn(10, 0, 2))
        res = gulinalg.update_rank1(a, b, c)
        assert res.shape == (10, 0, 2)
        ref = np.stack([np.dot(np.zeros((0, 0)), np.zeros((0, 2))) + c[i]
                        for i in range(len(c))])
        assert_allclose(res, ref)

    def test_size_one_vector(self):
        """Test broadcasting for vector inputs of size one"""
        a = np.ascontiguousarray(np.random.randn(10, 1))
        b = np.ascontiguousarray(np.random.randn(10, 1))
        c = np.ascontiguousarray(np.random.randn(10, 1, 1))
        res = gulinalg.update_rank1(a, b, c)
        assert res.shape == (10, 1, 1)
        ref = np.stack([np.dot(a[i].reshape(1, 1), b[i].reshape(1, 1)) + c[i]
                        for i in range(len(c))])
        assert_allclose(res, ref)


class TestSyrk(TestCase):

    def test_syrk_zeros_c(self):
        gufunc = partial(gulinalg.update_rankk, sym_out=False)
        for a_trans in [True, False]:
            for dtype in [np.float32, np.float64]:
                a = np.array([[1., 0.],
                              [0., -2.],
                              [2., 3.]], dtype=dtype)
                if a_trans:
                    a = a.T
                c = np.zeros((a.shape[0],)*2, dtype=dtype)

                expected = np.dot(a, a.T) + c

                # test upper triangular case
                r = gufunc(a, c, transpose_type='N', UPLO='U')
                assert_allclose(np.triu(expected), r)
                assert_(r.dtype == a.dtype)

                # test lower triangular case
                r = gufunc(a, c, transpose_type='N', UPLO='L')
                assert_allclose(np.tril(expected), r)
                assert_(r.dtype == a.dtype)

                # test upper triangular case with transpose_type='T'
                r = gufunc(a.T, c, transpose_type='T', UPLO='U')
                assert_allclose(np.triu(expected), r)
                assert_(r.dtype == a.dtype)

                # test lower triangular case with transpose_type='T'
                r = gufunc(a.T, c, transpose_type='T', UPLO='L')
                assert_allclose(np.tril(expected), r)
                assert_(r.dtype == a.dtype)

    def test_syrk_ones_c(self):
        gufunc = partial(gulinalg.update_rankk, sym_out=False)
        for a_trans in [True, False]:
            for dtype in [np.float32, np.float64]:
                a = np.array([[1., 0.],
                              [0., -2.],
                              [2., 3.]], dtype=dtype)
                if a_trans:
                    a = a.T
                c = np.ones((a.shape[0],)*2, dtype=dtype)
                tmp = np.dot(a, a.T) + c
                mask_upper = np.triu(c) > 0
                mask_lower = np.tril(c) > 0
                expected_lower = tmp.copy()
                expected_lower[~mask_lower] = c[~mask_lower]
                expected_upper = tmp.copy()
                expected_upper[~mask_upper] = c[~mask_upper]

                # test upper triangular case
                r = gufunc(a, c, transpose_type='N', UPLO='U')
                assert_allclose(expected_upper, r)
                assert_(r.dtype == a.dtype)

                # test lower triangular case
                r = gufunc(a, c, transpose_type='N', UPLO='L')
                assert_allclose(expected_lower, r)
                assert_(r.dtype == a.dtype)

                # test upper triangular case, with transpose_type='T'
                r = gufunc(a.T, c, transpose_type='T', UPLO='U')
                assert_allclose(expected_upper, r)
                assert_(r.dtype == a.dtype)

                # test lower triangular case, with transpose_type='T'
                r = gufunc(a.T, c, transpose_type='T', UPLO='L')
                assert_allclose(expected_lower, r)
                assert_(r.dtype == a.dtype)

    def test_syrk_broadcasted(self):
        for sym_out, workers in product([False, True], [1, -1]):
            gufunc = partial(gulinalg.update_rankk, sym_out=sym_out,
                             workers=workers)
            for a_trans in [True, False]:
                for dtype in [np.float32, np.float64]:
                    a = np.array([[1., 0.],
                                  [0., -2.],
                                  [2., 3.]], dtype=dtype)
                    if a_trans:
                        a = a.T
                    c = np.zeros((a.shape[0],)*2, dtype=dtype)
                    expected = np.dot(a, a.swapaxes(-1, -2)) + c

                    a = np.stack((a, ) * n_batch, axis=0)  # stack

                    # test upper triangular case
                    r = gufunc(a, c, transpose_type='N', UPLO='U')
                    for i in range(n_batch):
                        if sym_out:
                            assert_allclose(expected, r[i])
                        else:
                            assert_allclose(np.triu(expected), r[i])
                    assert_(r.dtype == a.dtype)

                    # test upper triangular case with extra c dimensions
                    c_4d = c[np.newaxis, np.newaxis, ...]
                    r = gufunc(a, c_4d, transpose_type='N', UPLO='U')
                    for i in range(n_batch):
                        if sym_out:
                            assert_allclose(expected, r[0][i])
                        else:
                            assert_allclose(np.triu(expected), r[0][i])
                    assert_(r.dtype == a.dtype)

                    # test lower triangular case
                    r = gufunc(a, c, transpose_type='N', UPLO='L')
                    for i in range(n_batch):
                        if sym_out:
                            assert_allclose(expected, r[i])
                        else:
                            assert_allclose(np.tril(expected), r[i])
                    assert_(r.dtype == a.dtype)

                    # test upper triangular case with transpose_type='T'
                    r = gufunc(a.swapaxes(-1, -2), c, transpose_type='T',
                               UPLO='U')
                    for i in range(n_batch):
                        if sym_out:
                            assert_allclose(expected, r[i])
                        else:
                            assert_allclose(np.triu(expected), r[i])
                    assert_(r.dtype == a.dtype)

                    # test lower triangular case
                    r = gufunc(a, c, transpose_type='N', UPLO='L')
                    for i in range(n_batch):
                        if sym_out:
                            assert_allclose(expected, r[i])
                        else:
                            assert_allclose(np.tril(expected), r[i])
                    assert_(r.dtype == a.dtype)

                    # test upper triangular case with transpose_type='T'
                    r = gufunc(a.swapaxes(-1, -2), c, transpose_type='T',
                               UPLO='U')
                    for i in range(n_batch):
                        if sym_out:
                            assert_allclose(expected, r[i])
                        else:
                            assert_allclose(np.triu(expected), r[i])
                    assert_(r.dtype == a.dtype)

                    # test lower triangular case with transpose_type='T'
                    r = gufunc(a.swapaxes(-1, -2), c, transpose_type='T',
                               UPLO='L')
                    for i in range(n_batch):
                        if sym_out:
                            assert_allclose(expected, r[i])
                        else:
                            assert_allclose(np.tril(expected), r[i])

                    assert_(r.dtype == a.dtype)

    def test_syrk_no_c_broadcasted(self):
        for sym_out, workers in product([False, True], [1, -1]):
            gufunc = partial(gulinalg.update_rankk, sym_out=sym_out,
                             workers=workers)
            for a_trans in [True, False]:
                for dtype in [np.float32, np.float64]:
                    a = np.array([[1., 0.],
                                  [0., -2.],
                                  [2., 3.]], dtype=dtype)
                    if a_trans:
                        a = a.T
                    c = np.zeros((a.shape[0],)*2, dtype=dtype)
                    expected = np.dot(a, a.swapaxes(-1, -2)) + c

                    a = np.stack((a, ) * n_batch, axis=0)  # stack

                    # test upper triangular case
                    r = gufunc(a, transpose_type='N', UPLO='U')
                    for i in range(n_batch):
                        if sym_out:
                            assert_allclose(expected, r[i])
                        else:
                            assert_allclose(np.triu(expected), r[i])
                    assert_(r.dtype == a.dtype)

                    # test lower triangular case
                    r = gufunc(a, transpose_type='N', UPLO='L')
                    for i in range(n_batch):
                        if sym_out:
                            assert_allclose(expected, r[i])
                        else:
                            assert_allclose(np.tril(expected), r[i])
                    assert_(r.dtype == a.dtype)

                    # test upper triangular case with transpose_type='T'
                    r = gufunc(a.swapaxes(-1, -2), transpose_type='T',
                               UPLO='U')
                    for i in range(n_batch):
                        if sym_out:
                            assert_allclose(expected, r[i])
                        else:
                            assert_allclose(np.triu(expected), r[i])
                    assert_(r.dtype == a.dtype)

                    # test lower triangular case
                    r = gufunc(a, transpose_type='N', UPLO='L')
                    for i in range(n_batch):
                        if sym_out:
                            assert_allclose(expected, r[i])
                        else:
                            assert_allclose(np.tril(expected), r[i])
                    assert_(r.dtype == a.dtype)

    def test_syrk_wrong_shape(self):
        for a_trans in [True, False]:
            for dtype in [np.float32, np.float64]:
                a = np.array([[1., 0.],
                              [0., -2.],
                              [2., 3.]], dtype=dtype)
                if a_trans:
                    a = a.T

                # use wrong axis size for c when transpose_type='N'
                c = np.zeros((a.shape[-1],)*2, dtype=dtype)

                with assert_raises(ValueError):
                    gulinalg.update_rankk(a, c, transpose_type='N')

                # use wrong axis size for c when transpose_type='T'
                c = np.zeros((a.shape[0],)*2, dtype=dtype)

                with assert_raises(ValueError):
                    gulinalg.update_rankk(a, c, transpose_type='T')

    def test_syrk_wrong_dtype(self):
        for dtype in [np.complex64, np.complex128]:
            a = np.ones((3, 3), dtype=dtype)
            c = np.zeros_like(a)

            with assert_raises(NotImplementedError):
                gulinalg.update_rankk(a, c)

    def test_syrk_invalid_uplo(self):
        a = np.ones((3, 3), dtype=np.float64)
        c = np.zeros_like(a)

        with assert_raises(ValueError):
            gulinalg.update_rankk(a, c, UPLO='X')

    def test_syrk_invalid_transpose_type(self):
        a = np.ones((3, 3), dtype=np.float64)
        c = np.zeros_like(a)

        with assert_raises(ValueError):
            gulinalg.update_rankk(a, c, transpose_type='X')

    def test_syrk_invalid_workers(self):
        a = np.ones((3, 3), dtype=np.float64)

        with assert_raises(ValueError):
            gulinalg.update_rankk(a, workers=0)


class TestQuadraticForm(TestCase):

    """Tests for Quadratic Form u * Q * v"""
    def test_quadratic_c(self):
        """test vectorized matrix multiply"""
        u = np.random.randn(M)
        q = np.random.randn(M, N)
        v = np.random.randn(N)
        res = gulinalg.quadratic_form(u, q, v)
        ref = np.dot(u[np.newaxis, :], np.dot(q, v[:, np.newaxis]))
        assert_allclose(res, ref)

    def test_quadratic_broadcast_u(self):
        """test vectorized matrix multiply"""
        u = np.random.randn(n_batch, M)
        q = np.random.randn(M, N)
        v = np.random.randn(N)
        for workers in [1, -1]:
            res = gulinalg.quadratic_form(u, q, v, workers=workers)
            ref = [np.squeeze(np.dot(u[i:i+1, :], np.dot(q, v[:, np.newaxis])))
                   for i in range(n_batch)]
            assert_allclose(res, ref)

    def test_quadratic_broadcast_uv(self):
        """test vectorized matrix multiply"""
        u = np.random.randn(n_batch, M)
        q = np.random.randn(M, N)
        v = np.random.randn(n_batch, N)
        for workers in [1, -1]:
            res = gulinalg.quadratic_form(u, q, v, workers=workers)
            ref = [np.squeeze(np.dot(u[i:i+1, :],
                                     np.dot(q, v[i, :, np.newaxis])))
                   for i in range(n_batch)]
            assert_allclose(res, ref)

    def test_quadratic_broadcast_uqv(self):
        """test vectorized matrix multiply"""
        u = np.random.randn(n_batch, M)
        q = np.random.randn(n_batch, M, N)
        v = np.random.randn(n_batch, N)
        for workers in [1, -1]:
            res = gulinalg.quadratic_form(u, q, v, workers=workers)
            ref = [np.squeeze(np.dot(u[i:i+1, :],
                                     np.dot(q[i, ...], v[i, :, np.newaxis])))
                   for i in range(n_batch)]
            assert_allclose(res, ref)

    def test_quadratic_broadcast_uqv_contiguity(self):
        """test vectorized matrix multiply"""
        u = np.asfortranarray(np.random.randn(n_batch, M))
        q = np.random.randn(n_batch, M, N, 2)[..., 0]  # non-contiguous
        v = np.random.randn(n_batch, N)
        for workers in [1, -1]:
            res = gulinalg.quadratic_form(u, q, v, workers=workers)
            ref = [np.squeeze(np.dot(u[i:i+1, :],
                                     np.dot(q[i, ...], v[i, :, np.newaxis])))
                   for i in range(n_batch)]
            assert_allclose(res, ref)

