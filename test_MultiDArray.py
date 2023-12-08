import pytest
import _MultiDArray

def test_InitializeZero():
    dim = 5
    mat1 = _MultiDArray.MultiDArray(dim, dim)
    AllZero = True
    for r in range(dim):
        for c in range(dim):
            if mat1[r, c] != 0.0:
                AllZero = False
                break
        if AllZero != True:
            break
    assert AllZero

def test_InitializeVector():
    dim = 2
    mat2 = _MultiDArray.MultiDArray(dim, dim, [0.0, 0.1, 1.0, 1.1])
    assert mat2[0, 0] == 0.0
    assert mat2[0, 1] == 0.1
    assert mat2[1, 0] == 1.0
    assert mat2[1, 1] == 1.1

def test_SameResult():
    dim = 1200
    mat1 = _MultiDArray.MultiDArray(dim, dim)
    mat2 = _MultiDArray.MultiDArray(dim, dim)
    for i in range(mat1.nrow):
        for j in range(mat1.ncol):
            mat1[i, j] = i * dim + j
            mat2[i, j] = 1
    ret_tile = _MultiDArray.multiply_tile(mat1, mat2, 128)
    ret_naive = _MultiDArray.multiply_naive(mat1, mat2)

    assert ret_tile == ret_naive


if __name__ == '__main__':
    pytest.main()
