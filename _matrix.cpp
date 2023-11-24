#include <iostream>
#include <iomanip>
#include <vector>
#include <stdexcept>
#include <cmath>
// #include <pybind11/stl.h>
// #include <pybind11/pybind11.h>
#include "mkl_cblas.h"

using namespace std;

class Matrix {

    public:

        Matrix(size_t nrow, size_t ncol)
        : m_nrow(nrow), m_ncol(ncol)
        {
            reset_buffer(nrow, ncol);
        }

        Matrix(size_t nrow, size_t ncol, vector<double> const & vec)
        : m_nrow(nrow), m_ncol(ncol)
        {
            reset_buffer(nrow, ncol);
            (*this) = vec;
        }

        Matrix & operator=(vector<double> const & vec)
        {
            if (size() != vec.size())
            {
                throw out_of_range("number of elements mismatch");
            }

            size_t k = 0;
            for (size_t i=0; i<m_nrow; ++i)
            {
                for (size_t j=0; j<m_ncol; ++j)
                {
                    (*this)(i,j) = vec[k];
                    ++k;
                }
            }

            return *this;
        }

        Matrix(Matrix const & other)
        : m_nrow(other.m_nrow), m_ncol(other.m_ncol)
        {
            reset_buffer(other.m_nrow, other.m_ncol);
            for (size_t i=0; i<m_nrow; ++i)
            {
                for (size_t j=0; j<m_ncol; ++j)
                {
                    (*this)(i,j) = other(i,j);
                }
            }
        }

        Matrix & operator=(Matrix const & other)
        {
            if (this == &other) {
                return *this;
            }
            if (m_nrow != other.m_nrow || m_ncol != other.m_ncol) {
                reset_buffer(other.m_nrow, other.m_ncol);
            }
            for (size_t i=0; i<m_nrow; ++i)
            {
                for (size_t j=0; j<m_ncol; ++j)
                {
                    (*this)(i,j) = other(i,j);
                }
            }
            return *this;
        }

        Matrix(Matrix && other)
        : m_nrow(other.m_nrow), m_ncol(other.m_ncol)
        {
            reset_buffer(0, 0);
            swap(m_nrow, other.m_nrow);
            swap(m_ncol, other.m_ncol);
            swap(m_buffer, other.m_buffer);
        }

        Matrix & operator=(Matrix && other)
        {
            if (this == &other) {
                return *this;
            }
            reset_buffer(0, 0);
            swap(m_nrow, other.m_nrow);
            swap(m_ncol, other.m_ncol);
            swap(m_buffer, other.m_buffer);
            return *this;
        }

        ~Matrix()   // return, clear buffer
        {
            reset_buffer(0, 0);
        }

        double   operator() (size_t row, size_t col) const
        {
            return m_buffer[index(row, col)];
        }
        double & operator() (size_t row, size_t col)
        {
            return m_buffer[index(row, col)];
        }

        bool operator== (Matrix const & other)
        {
            if ((m_ncol != other.ncol()) || (m_nrow != other.ncol()))
            {
                return false;
            }

            for (size_t i=0; i<m_nrow; ++i)
            {
                for (size_t j=0; j<m_ncol; ++j)
                {
                    if (m_buffer[index(i, j)] != other.m_buffer[other.index(i, j)])
                    {
                        return false;
                    }
                }
            }

            return true;
        }

        bool operator!= (Matrix const & other)
        {
            return !(*this == other);
        }

        size_t nrow() const {
            return m_nrow;
        }
        size_t ncol() const {
            return m_ncol;
        }

        size_t size() const {
            return m_nrow * m_ncol;
        }
        double  buffer(size_t i) const {
            return m_buffer[i];
        }
        double *buffer() const {
            return m_buffer;
        }
        std::vector<double> buffer_vector() const
        {
            return std::vector<double>(m_buffer, m_buffer+size());
        }

        Matrix multiply_naive(Matrix const & other)
        {
            // validate_multiplication
            if (this->ncol() != other.nrow()) {
                throw std::out_of_range(
                    "the number of first matrix column "
                    "differs from that of second matrix row");
            }

            Matrix ret(this->nrow(), other.ncol());

            for (size_t i=0; i<ret.nrow(); ++i)
            {
                for (size_t k=0; k<ret.ncol(); ++k)
                {
                    double index_sum = 0;
                    for (size_t j=0; j<this->ncol(); ++j)    // j<mat2.nrow()
                    {
                        // std::cout << i << j << k << std::endl;
                        index_sum += this->m_buffer[index(i, j)] * other(j,k);
                    }
                    ret(i,k) = index_sum;
                }
            }

            return ret;
        }


        // Vector-Matrix multiplication.
        Matrix operator* (std::vector<double> const & vec) {
            // std::cout << ">> operator* (std::vector<double> const & vec)" << std::endl;
        }

        // Tiled matrix-matrix multiplication.
        Matrix multiply_tile(Matrix const & other, size_t const tsize) {
            // std::cout << ">> multiply_tile(Matrix const & other, size_t const tsize)" << std::endl;
            // validate_multiplication
            if (this->ncol() != other.nrow()) {
                throw std::out_of_range(
                    "the number of first matrix column "
                    "differs from that of second matrix row");
            }

            Matrix ret(this->nrow(), other.ncol());

            // const size_t ntrow1 = std::ceil(mat1.nrow() / tsize);
            // const size_t ntcol1 = std::ceil(mat1.ncol() / tsize);
            // const size_t ntrow2 = std::ceil(mat2.nrow() / tsize);
            // const size_t ntcol2 = std::ceil(mat2.ncol() / tsize);
            // cout << ntrow1 << " x " << ntcol1 << ", " << ntrow1 << " x " << ntcol2  << endl;

            for (size_t ti_row=0; ti_row<this->nrow(); ti_row+=tsize)
            {
                for (size_t ti_col=0; ti_col<other.ncol(); ti_col+=tsize)
                {
                    for (size_t ti_j=0; ti_j<this->ncol(); ti_j+=tsize)
                    {
                        // within a tile
                        size_t tr_end = std::min(this->nrow(), ti_row + tsize);
                        size_t tc_end = std::min(this->ncol(), ti_col + tsize);
                        size_t tj_end = std::min(this->ncol(), ti_j   + tsize);
                        // std::cout << ti_row << " " << ti_j << " " << ti_col << ", " << tr_end << " " << tj_end << " " << tc_end << std::endl;
                        
                        for (size_t i=ti_row; i<tr_end; ++i)
                        {
                            for (size_t k=ti_col; k<tc_end; ++k)
                            {
                                for (size_t j=ti_j; j<tj_end; ++j)    // j<mat2.nrow()
                                {
                                    // std::cout << ret(i,k) << " " << i << j << k << " " << mat1(i,j) << " * " << mat2(j,k) << std::endl;
                                    ret(i,k) += m_buffer[index(i, j)] * other(j,k);
                                }
                            }
                        }
                    }
                }
            }

            return ret;
        }

        // Element-wise Matrix Multiplication
        Matrix elemwise(Matrix const & other) {
            // operator* (Matrix const & other)" << std::endl;
        }

        Matrix Sum(){};
        Matrix Mean(){};
        Matrix Var(){};
        Matrix Std(){};
        // Linear Self-Covariance Matrix Multiplication
        Matrix Cov(){};

    private:

        size_t index(size_t row, size_t col) const
        {
            return row + col * m_nrow;
        }

        void reset_buffer(size_t nrow, size_t ncol)
        {
            if (m_buffer) {
                delete[] m_buffer;
            }
            const size_t nelement = nrow * ncol;
            if (nelement) {
                m_buffer = new double[nelement]();
            }
            else {
                m_buffer = nullptr;
            }
            m_nrow = nrow;
            m_ncol = ncol;
        }

        size_t m_nrow = 0;
        size_t m_ncol = 0;
        double * m_buffer = nullptr;

};


void validate_multiplication(Matrix const & mat1, Matrix const & mat2)
{
    if (mat1.ncol() != mat2.nrow())
    {
        throw std::out_of_range(
            "the number of first matrix column "
            "differs from that of second matrix row");
    }
}

// Naive matrix-matrix multiplication.
Matrix multiply_naive(Matrix const & mat1, Matrix const & mat2)
{
    validate_multiplication(mat1, mat2);

    Matrix ret(mat1.nrow(), mat2.ncol());

    for (size_t i=0; i<ret.nrow(); ++i)
    {
        for (size_t k=0; k<ret.ncol(); ++k)
        {
            double index_sum = 0;
            for (size_t j=0; j<mat1.ncol(); ++j)    // j<mat2.nrow()
            {
                // std::cout << i << j << k << std::endl;
                index_sum += mat1(i,j) * mat2(j,k);
            }
            ret(i,k) = index_sum;
        }
    }

    return ret;
}

// Tiled matrix matrix multiplication.
Matrix multiply_tile(Matrix const & mat1, Matrix const & mat2, size_t const tsize)
{
    validate_multiplication(mat1, mat2);

    Matrix ret(mat1.nrow(), mat2.ncol());

    // const size_t ntrow1 = std::ceil(mat1.nrow() / tsize);
    // const size_t ntcol1 = std::ceil(mat1.ncol() / tsize);
    // const size_t ntrow2 = std::ceil(mat2.nrow() / tsize);
    // const size_t ntcol2 = std::ceil(mat2.ncol() / tsize);
    // cout << ntrow1 << " x " << ntcol1 << ", " << ntrow1 << " x " << ntcol2  << endl;

    for (size_t ti_row=0; ti_row<mat1.nrow(); ti_row+=tsize)
    {
        for (size_t ti_col=0; ti_col<mat2.ncol(); ti_col+=tsize)
        {
            for (size_t ti_j=0; ti_j<mat1.ncol(); ti_j+=tsize)
            {
                // within a tile
                size_t tr_end = std::min(mat1.nrow(), ti_row + tsize);
                size_t tc_end = std::min(mat2.ncol(), ti_col + tsize);
                size_t tj_end = std::min(mat1.ncol(), ti_j   + tsize);
                // std::cout << ti_row << " " << ti_j << " " << ti_col << ", " << tr_end << " " << tj_end << " " << tc_end << std::endl;
                
                for (size_t i=ti_row; i<tr_end; ++i)
                {
                    for (size_t k=ti_col; k<tc_end; ++k)
                    {
                        for (size_t j=ti_j; j<tj_end; ++j)    // j<mat2.nrow()
                        {
                            // std::cout << ret(i,k) << " " << i << j << k << " " << mat1(i,j) << " * " << mat2(j,k) << std::endl;
                            ret(i,k) += mat1(i,j) * mat2(j,k);
                        }
                    }
                }
            }
        }
    }

    return ret;
}

Matrix multiply_mkl(Matrix const & mat1, Matrix const & mat2)
{
    validate_multiplication(mat1, mat2);

    Matrix ret(mat1.nrow(), mat2.ncol());

    cblas_dgemm(
        CblasRowMajor   /* const CBLAS_LAYOUT Layout */
      , CblasNoTrans    /* const CBLAS_TRANSPOSE transa */
      , CblasNoTrans    /* const CBLAS_TRANSPOSE transb */
      , mat1.nrow()     /* const MKL_INT m */
      , mat2.ncol()     /* const MKL_INT n */
      , mat1.ncol()     /* const MKL_INT k */
      , 1.0             /* const double alpha */
      , mat1.buffer()   /* const double *a */
      , mat1.ncol()     /* const MKL_INT lda */
      , mat2.buffer()   /* const double *b */
      , mat2.ncol()     /* const MKL_INT ldb */
      , 0.0             /* const double beta */
      , ret.buffer()    /* double * c */
      , ret.ncol()      /* const MKL_INT ldc */
    );

    return ret;
}


std::ostream & operator << (std::ostream & ostr, Matrix const & mat)
{
    // std::cout << "operator << (print matrix)" << std::endl;
    for (size_t i=0; i<mat.nrow(); ++i)
    {
        ostr << std::endl << " ";
        for (size_t j=0; j<mat.ncol(); ++j)
        {
            ostr << " " << std::setw(2) << mat(i, j);
        }
    }

    return ostr;
}

int main(int argc, char ** argv)
{
    std::cout << ">>> A(2x3) times B(3x2):" << std::endl;
    Matrix mat1(2, 3, std::vector<double>{1, 2, 3, 4, 5, 6});
    Matrix mat2(3, 2, std::vector<double>{1, 2, 3, 4, 5, 6});

    std::cout << "matrix A (2x3):" << mat1 << std::endl;
    std::cout << "matrix B (3x2):" << mat2 << std::endl;

    Matrix ret_naive = multiply_naive(mat1, mat2);
    std::cout << "multiply_naive result matrix C (2x2) = AB:" << ret_naive << std::endl;

    Matrix ret_naive_com = mat1.multiply_naive(mat2);
    std::cout << "multiply_naive result matrix C (2x2) = AB:" << ret_naive_com << std::endl;

    Matrix ret_mkl = multiply_mkl(mat1, mat2);
    std::cout << "multiply_mkl result matrix C (2x2) = AB:" << ret_mkl << std::endl;

    return 0;
}


// PYBIND11_MODULE(_matrix, m) {   // module name
//     m.doc() = "Matrix-Matrix Multiplication.";

//     pybind11::class_<Matrix>(m, "Matrix")
//         .def(pybind11::init<size_t, size_t>())
//         .def(pybind11::init<size_t, size_t, std::vector<double> const & >())
//         .def("__getitem__",
//             [](const Matrix& mat, std::pair<size_t, size_t> index) {
//                 return mat(index.first, index.second);
//             }
//         )
//         .def("__setitem__",
//             [](Matrix& mat, std::pair<size_t, size_t> index, double value) {
//                 mat(index.first, index.second) = value;
//             }
//         )
//         .def("__eq__", &Matrix::operator==)
//         .def("__ne__", &Matrix::operator!=)
//         .def_property_readonly("nrow", &Matrix::nrow)
//         .def_property_readonly("ncol", &Matrix::ncol);

//     m.def("multiply_naive", &multiply_naive, pybind11::arg("mat1"), pybind11::arg("mat2"));
//     m.def("multiply_tile", &multiply_tile, pybind11::arg("mat1"), pybind11::arg("mat2"), pybind11::arg("tsize"));
//     m.def("multiply_mkl", &multiply_mkl, pybind11::arg("mat1"), pybind11::arg("mat2"));
// }
