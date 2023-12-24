import pytest
import _MultiDArray as mdarr

def test_Initialize():
    shape0 = [5]
    arr0 = mdarr.MultiDArrayFloat32(shape0)
    assert arr0.dim == 1
    assert arr0.shape == [5]
    for dim0 in range(shape0[0]):
        arr0[dim0,] = 0
        
    for dim0 in range(shape0[0]):
        assert arr0[dim0,] == 0.0

    shape1 = [2, 2, 3]
    arr0 = mdarr.MultiDArrayInt64(shape1, 0)
    arr1 = mdarr.MultiDArrayInt64(shape1)
    assert arr0.dim == arr1.dim == 3
    assert arr0.shape == arr1.shape == [2, 2, 3]
    for dim0 in range(shape1[0]):
        for dim1 in range(shape1[1]):
            for dim2 in range(shape1[2]):
                arr1[dim0, dim1, dim2] = 0

    for dim0 in range(shape1[0]):
        for dim1 in range(shape1[1]):
            for dim2 in range(shape1[2]):
                assert arr0[dim0, dim1, dim2] == 0
    assert arr0 == arr1
    
    arr0 = mdarr.MultiDArrayFloat32(shape1, 0.0)
    arr1 = mdarr.MultiDArrayFloat32(shape1)
    assert arr0.dim == arr1.dim == 3
    assert arr0.shape == arr1.shape == [2, 2, 3]
    for dim0 in range(shape1[0]):
        for dim1 in range(shape1[1]):
            for dim2 in range(shape1[2]):
                arr1[dim0, dim1, dim2] = 0.0

    for dim0 in range(shape1[0]):
        for dim1 in range(shape1[1]):
            for dim2 in range(shape1[2]):
                assert arr1[dim0, dim1, dim2] == 0.0
    assert arr0 == arr1

def test_elementwise_multiply():
    shape1 = [2, 2, 3]
    shape2 = [2, 2, 3]

    arr1 = mdarr.MultiDArrayInt64(shape1, 2)
    arr2 = mdarr.MultiDArrayInt64(shape2, 3)
    arr3 = mdarr.MultiDArrayInt64(shape2, 4)

    arr1[0, 0, 1] = 1

    result_comp = arr1.elementwise_multiply(arr2).elementwise_multiply(arr3)
    result_fund = mdarr.elementwise_multiply(mdarr.elementwise_multiply(arr1, arr2), arr3)
    assert result_comp == result_fund

    for dim0 in range(shape1[0]):
        for dim1 in range(shape1[1]):
            for dim2 in range(shape1[2]):
                if dim0 == 0 and dim1 == 0 and dim2 == 1:
                    assert result_comp[dim0, dim1, dim2] == 12
                    assert result_fund[dim0, dim1, dim2] == 12
                else:
                    assert result_comp[dim0, dim1, dim2] == 24
                    assert result_fund[dim0, dim1, dim2] == 24

def test_sum():
    shape = [2, 2, 3]
    arr = mdarr.MultiDArrayInt64(shape, 2)
    arr[0, 0, 1] = 1
    arr[0, 0, 2] = 1

    result_comp = arr.sum()
    assert result_comp == 22
    # result_fund = mdarr.sumInt64(arr)
    # assert type(result_comp) == type(result_fund) == int
    # assert result_comp == result_fund

    print(type(arr))
    result_comp = arr.sum(0)
    result_fund = mdarr.sum(arr, 0)
    assert result_comp.dim == result_fund.dim == 3
    assert result_comp.shape == result_fund.shape == [1, 2, 3]
    for dim0 in range(result_comp.shape[0]):
        for dim1 in range(result_comp.shape[1]):
            for dim2 in range(result_comp.shape[2]):
                if dim1 == 0 and (dim2 == 1 or dim2 == 2):
                    assert result_comp[dim0, dim1, dim2] == 3
                    assert result_fund[dim0, dim1, dim2] == 3
                else:
                    assert result_comp[dim0, dim1, dim2] == 4
                    assert result_fund[dim0, dim1, dim2] == 4
    assert result_comp == result_fund

    result_comp = arr.sum(1)
    result_fund = mdarr.sum(arr, 1)
    assert result_comp.dim == result_fund.dim == 3
    assert result_comp.shape == result_fund.shape == [2, 1, 3]
    for dim0 in range(result_comp.shape[0]):
        for dim1 in range(result_comp.shape[1]):
            for dim2 in range(result_comp.shape[2]):
                if dim0 == 0 and (dim2 == 1 or dim2 == 2):
                    assert result_comp[dim0, dim1, dim2] == 3
                    assert result_fund[dim0, dim1, dim2] == 3
                else:
                    assert result_comp[dim0, dim1, dim2] == 4
                    assert result_fund[dim0, dim1, dim2] == 4
    assert result_comp == result_fund

    result_comp = arr.sum(2)
    result_fund = mdarr.sum(arr, 2)
    assert result_comp.dim == result_fund.dim == 3
    assert result_comp.shape == result_fund.shape == [2, 2, 1]
    for dim0 in range(result_comp.shape[0]):
        for dim1 in range(result_comp.shape[1]):
            for dim2 in range(result_comp.shape[2]):
                if dim0 == 0 and dim1 == 0:
                    assert result_comp[dim0, dim1, dim2] == 4
                    assert result_fund[dim0, dim1, dim2] == 4
                else:
                    assert result_comp[dim0, dim1, dim2] == 6
                    assert result_fund[dim0, dim1, dim2] == 6
    assert result_comp == result_fund


if __name__ == '__main__':
    test_sum()
    # pytest.main()
