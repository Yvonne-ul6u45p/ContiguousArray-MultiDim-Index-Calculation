# Contiguous Array adding Multi-Dimensional Index Calculation


## Basic Information

Multi-dimensional arrays of fundamental types and struct are a building block
for numerical code. It may be as simple as a pointer to a contiguous memory
buffer, or well-designed meta-data with the memory buffer.

- My GitHub Repository:
    https://github.com/Yvonne-ul6u45p/ContiguousArray-MultiDim-Index-Calculation


## Problem to Solve

While a mere pointer works well with one-dimensional arrays, calculating the
pointer offset for multi-dimensional arrays makes the code for numerical
calculation cryptic and hard to maintain. It is very helpful to wrap the multi-
dimensional index calculation in a library.


## Prospective Users

This project targets users who search for an easy-handle multi-dimensional
array calculation using contiguous array.


## System Architecture

### Calculations

1. Multi-Dimension Array & Multi-Dimension Array Multiplication
1. Multi-Dimension Array & Multi-Dimension Array Multiplication with tiling
1. Element-wise Multi-Dimension Array Multiplication
1. Linear Self-Covariance Matrix of a Multi-Dimensional Array: $Cov(X, X) = \mathbb E[(X- \mathbb E[X])(X- \mathbb E[X])^T]$

* Additionl Function: add variables for specifying dimension


## API Description

### C++

#### Fundamental Types

```C++
// Naive Multi-Dimension Array & Multi-Dimension Array Multiplication.
MultiDArray operator* (MultiDArray const & mat1, MultiDArray const & mat2)
{
    // ...
}

// Tiled Multi-Dimension Array & Multi-Dimension Array Multiplication.
MultiDArray multiply_tile(MultiDArray const & mat1, MultiDArray const & mat2, size_t const tsize)
{
    // ...
}

// Element-wise Multi-Dimension Array Multiplication.
MultiDArray elem_multiply(MultiDArray const & mat1, MultiDArray const & mat2)
{
    // ...
}

// Linear Self-Covariance Matrix of Multi-Dimension Array.
MultiDArray linear_CovMat(MultiDArray const & mat1)
{
    // ...
}

```


#### Composite Types (struct)

```C++
struct MultiDArray
{
    public:

        MultiDArray(std::vector<size_t>);
        MultiDArray(std::vector<size_t>, std::vector<double> const & vec);
        MultiDArray & operator=(std::vector<double> const & vec)
        MultiDArray(MultiDArray const & other);
        MultiDArray & operator=(MultiDArray const & other);
        MultiDArray(MultiDArray && other);
        MultiDArray & operator=(MultiDArray && other);
        ~MultiDArray();
        double   operator() (std::vector<size_t>) const;
        double & operator() (std::vector<size_t>);
        bool operator== (MultiDArray const & mat2);
        bool operator!= (MultiDArray const & mat2);
        size_t dim(size_t d) const;
        size_t size() const;
        double  buffer(size_t i) const;
        std::vector<double> buffer_vector() const;

        // Naive Array-Array multiplication.
        MultiDArray operator@ (MultiDArray const & other)
        // Tiled Array-Array multiplication.
        MultiDArray multiply_tile(MultiDArray const & other, size_t const tsize);
        // Element-wise Array Multiplication
        MultiDArray operator* (MultiDArray const & other);        // element-wise multiplication
        // Linear Self-Covariance Matrix of Array
        MultiDArray Cov();
    
    private:

        size_t index(size_t row, size_t col) const;
        void reset_buffer(size_t nrow, size_t ncol);
        size_t m_nrow = 0;
        size_t m_ncol = 0;
        double * m_buffer = nullptr;
};
```

* The C++ user can link the library by compiling with:
    `g++ main.cpp -o -lcontiguous_matrix_calculate`


### Python
* The python API can accept the numpy ndarray format as inputs, and call C++
    API using pybind11.
* Users can use the library by adding:
    `import contiguous_array_calculate`


### Overall Architecture

```mermaid
```

## Engineering Infrastructure

1. Automatic build system: 
    - `CMake`
2. Version control:
    - `git`
3. Testing framework:
    - `pytest`
4. Documentation:
    - `Markdown`
5. Continuous Integration: 
    - Github Actions


## Schedule

* Write down encountered issues as much as possible.

* Week 1 (11/12):
    - Write `pytest`
    - Write Continuous Integration
* Week 2 (11/19):
    - C++: Implement *Element-wise Multi-Dimension Array Multiplication* in
            fundamental types
* Week 3 (11/26):
    - C++: Implement *Linear Self-Covariance Matrix Multiplication* in
            fundamental types
* Week 4 (12/03):
    - C++: Implement *composite types (struct)*
* Week 5 (12/10):
    - wrap up with `pybind`
    - Write Python API
* Week 6 (12/17):
    - Measure Performance, including runtime and memory consumption
    - write automatic build system by `Cmake`
* Week 7 (12/23):
    - Make Slides
    - Prepare for the presentation


## References

1. [solvcon / modmesh, Actions](https://github.com/solvcon/modmesh/actions/runs/6983552231/job/19004850729)
2. [contiguous container library - arrays](https://github.com/foonathan/array)
3. [contiguous array type](https://github.com/andrewthad/contiguous)
