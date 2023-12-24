
#include "SimpleArray.hpp"
#include "small_vector.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <initializer_list>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
// #include "mkl_cblas.h"

using namespace std;

// namespace modmesh
// {

template <typename T>
class MultiDArray : public SimpleArray<T>
{

public:
    using SimpleArray<T>::SimpleArray;
    using shape_type = small_vector<size_t>;

    // Element-wise multiplication function
    MultiDArray<T> elementwise_multiply(const MultiDArray<T> &other) const
    {
        if (!(this->shape() == other.shape()))
        {
            throw std::invalid_argument("Shapes of MultiDArrays must match for element-wise multiplication.");
        }

        MultiDArray<T> result(this->shape());

        for (size_t i = 0; i < this->size(); ++i)
        {
            result[i] = this->data(i) * other.data(i);
        }

        return result;
    }

    T sum() const {
        // Sum over all dimensions by calling sum with each dimension.
        auto result = 0;
        for (size_t i = 0; i < this->size(); ++i) {
            result += this->data(i);
        }
        return result;
    }
    MultiDArray<T> sum(size_t dim) const
    {
        if (dim >= this->ndim())
        {
            throw std::out_of_range("Invalid dimension for sum operation.");
        }

        small_vector<size_t> new_shape = this->shape();
        new_shape[dim] = 1;
        MultiDArray<T> result(new_shape, 0);

        for (size_t i = 0; i < this->size(); ++i)
        {
            // Compute the corresponding index in the result array.
            small_vector<size_t> idx_new = this->calculate_index(i);
            idx_new[dim] = 0;
            result.at(idx_new) += this->data(i);
        }

        return result;
    }

    shape_type calculate_index(size_t index) const
    {
        shape_type idx_new(this->ndim());
        for (size_t i = 0; i < this->ndim(); ++i)
        {
            idx_new[i] = index / this->stride(i);
            index -= idx_new[i] * this->stride(i);
        }
        return idx_new;
    }

    bool operator==(const MultiDArray<T> &other)
    {
        if (!(this->shape() == other.shape()))
        {
            return false;
        }

        for (size_t i = 0; i < this->size(); ++i)
        {
            if (this->data(i) != other.data(i))
            {
                return false;
            }
        }

        return true;
    }

    bool operator!=(const MultiDArray<T> &other)
    {
        return !(*this == other);
    }
};

template <typename T>
MultiDArray<T> elementwise_multiply(MultiDArray<T> const &arr1, MultiDArray<T> const &arr2)
{
    if (!(arr1.shape() == arr2.shape()))
    {
        throw std::invalid_argument("Shapes of MultiDArrays must match for element-wise multiplication.");
    }

    MultiDArray<T> result(arr1.shape());

    for (size_t i = 0; i < arr1.size(); ++i)
    {
        result[i] = arr1.data(i) * arr2.data(i);
    }

    return result;
}


// template <typename T>
// T sum(MultiDArray<T> const &arr) {
//     // Sum over all dimensions by calling sum with each dimension.
//     auto result = 0;
//     for (size_t i = 0; i < arr.size(); ++i) {
//         result += arr.data(i);
//     }
//     return result;
// }
template <typename T>
MultiDArray<T> sum(MultiDArray<T> const &arr, size_t dim)
{
    if (dim >= arr.ndim())
    {
        throw std::out_of_range("Invalid dimension for sum operation.");
    }

    small_vector<size_t> new_shape = arr.shape();
    new_shape[dim] = 1; // The size of the summed dimension will be reduced to 1.
    MultiDArray<T> result(new_shape, 0);

    for (size_t i = 0; i < arr.size(); ++i)
    {
        // Compute the corresponding index in the result array.
        small_vector<size_t> idx_new = arr.calculate_index(i);
        idx_new[dim] = 0;
        result.at(idx_new) += arr.data(i);
    }

    return result;
}


template <typename T>
std::ostream &print_recursive(std::ostream &ostr, const MultiDArray<T> &arr, std::vector<size_t> indices, size_t dim)
{
    if (dim == arr.ndim())
    {
        // Reached the last dimension, print the element.
        ostr << arr.at(indices) << " ";
    }
    else
    {
        // Iterate over the current dimension and recurse to the next one.
        size_t size = arr.shape(dim);
        ostr << std::endl;
        for (size_t i = 0; i < size; ++i)
        {
            indices[dim] = i;
            print_recursive(ostr, arr, indices, dim + 1);
        }
        // ostr << "]";
    }
    return ostr;
}

template <typename T>
std::ostream &operator<<(std::ostream &ostr, const MultiDArray<T> &arr)
{
    const auto &shape = arr.shape();
    std::vector<size_t> indices(shape.size(), 0);
    return print_recursive(ostr, arr, indices, 0);
}

template <typename T, size_t N>
std::ostream &operator<<(std::ostream &ostr, const small_vector<T, N> &shape)
{
    ostr << "small_vector: (";
    for (size_t i = 0; i < shape.size(); ++i)
    {
        ostr << shape[i];
        if (i < shape.size() - 1)
        {
            ostr << ", ";
        }
    }
    ostr << ")";
    return ostr;
}

int main(int argc, char **argv)
{
    small_vector<size_t> shape_vector1 = {2, 2, 3};
    small_vector<size_t> shape_vector2 = {2, 2, 3};

    MultiDArray<int> arr1(shape_vector1, 2);
    MultiDArray<int> arr2(shape_vector2, 3);
    MultiDArray<int> arr3(shape_vector2, 4);

    arr1(0, 0, 0) = 1;
    arr1(0, 0, 1) = 1;

    MultiDArray<int> result = sum(arr1, 2);
    std::cout << result << endl;
    result = arr1.sum(2);
    std::cout << result << endl;
    int resultint = arr1.sum();
    std::cout << resultint << endl;
    // resultint = sum(arr1);
    // std::cout << resultint << endl;

    MultiDArray<int> result_comp = arr1.elementwise_multiply(arr2).elementwise_multiply(arr3);
    MultiDArray<int> result_fund = elementwise_multiply(elementwise_multiply(arr1, arr2), arr3);

    std::cout << result_comp << std::endl;
    std::cout << result_fund << std::endl;
    std::cout << (arr1 == arr2) << std::endl;
    std::cout << (arr2 == arr2) << std::endl;
    std::cout << (arr2 != arr3) << std::endl;
    std::cout << (arr3 != arr3) << std::endl;

    return 0;
}

namespace py = pybind11;

template <typename T>
void bind_multidarray(py::module &m, const char *typestr) {
    py::class_<MultiDArray<T>>(m, ("MultiDArray" + std::string(typestr)).c_str(), py::buffer_protocol())
        .def(py::init<size_t>(), py::arg("length"))
        .def(py::init<std::vector<size_t> const & >(), py::arg("shape"))
        .def(py::init<std::vector<size_t> const &, T>(), py::arg("shape"), py::arg("value"))
        .def(py::init<const small_vector<size_t>&>())
        .def(py::init<const small_vector<size_t>&, T>())
        .def("__getitem__",
            [](const MultiDArray<T> &arr, py::tuple indices) {
                if (indices.size() != arr.ndim()) {
                    throw std::invalid_argument("Number of indices must match the number of dimensions.");
                }

                // Convert Python indices to C++ vector
                std::vector<size_t> cxx_indices;
                for (size_t i = 0; i < indices.size(); ++i) {
                    cxx_indices.push_back(indices[i].cast<size_t>());
                }

                // Use at function with the converted indices
                return arr.at(cxx_indices);
            }
        )
        .def("__setitem__",
            [](MultiDArray<T> &arr, py::tuple indices, T value) {
                if (indices.size() != arr.ndim()) {
                    throw std::invalid_argument("Number of indices must match the number of dimensions.");
                }

                // Convert Python indices to C++ vector
                std::vector<size_t> cxx_indices;
                for (size_t i = 0; i < indices.size(); ++i) {
                    cxx_indices.push_back(indices[i].cast<size_t>());
                }

                // Use at function with the converted indices
                arr.at(cxx_indices) = value;
            }
        )
        .def("elementwise_multiply", &MultiDArray<T>::elementwise_multiply, py::arg("other"))
        .def("sum", [](const MultiDArray<T>& self) -> T { return self.sum(); }, "Sum for MultiDArray")
        .def("sum", [](const MultiDArray<T>& self, size_t dim) -> MultiDArray<T> { return self.sum(dim); }, "Sum for MultiDArray")
        .def("__eq__", &MultiDArray<T>::operator==)
        .def("__ne__", &MultiDArray<T>::operator!=)
        .def_property_readonly("shape",
            [typestr](MultiDArray<T>& mdarr) -> py::list {
                // Convert small_vector to Python list
                py::list result;
                for (const auto& element : mdarr.shape()) {
                    result.append(element);
                }
                return result;
            }
        )
        .def_property_readonly("dim",
            [](MultiDArray<T>& mdarr) -> size_t {
                return mdarr.shape().size();
            }
        );
}

PYBIND11_MODULE(_MultiDArray, m) {
    m.doc() = "MultiDArray & MultiDArray Calculation.";

    // Bind MultiDArray for different data types
    bind_multidarray<int64_t>(m, "Int64");
    bind_multidarray<float>(m, "Float32");

    m.def("elementwise_multiply", &elementwise_multiply<int64_t>, "Element-wise multiplication for MultiDArray");
    m.def("elementwise_multiply", &elementwise_multiply<float>, "Element-wise multiplication for MultiDArray");
    m.def("sum", &sum<int64_t>, "Sum for MultiDArray");
    m.def("sum", &sum<float>, "Sum for MultiDArray");
    // m.def("sumInt64", py::overload_cast<const MultiDArray<int64_t>&>(&sum<int64_t>), "Sum for MultiDArray (int64_t version)");
    // m.def("sumFloat32", py::overload_cast<const MultiDArray<float>&>(&sum<float>), "Sum for MultiDArray (float version)");

}

