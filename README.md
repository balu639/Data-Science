# Data-Science
A complete Data Science tutorial.

This Tutorial explains the following introductory concepts to Data Science tools from the basics.

## Python
A Programming language is required to build the data science codes, and the best ,effortless programming language in the market is Python. In this tutorial , we will go through the following python programming concepts.

### 1.Variables
### 2.Numbers
### 3.Strings
### 4.Lists
### 5.Conditional Statements
### 6.Loops
### 7.Functions
### 8.Dictionaries 
### 9.Tuples
### 10.Working with Json
### 11.Exception Handling
### 12. Classes and Objects

## Numpy
Numpy is a powerful Python library for numerical computing. It provides a multidimensional array object, various mathematical functions, and tools for working with arrays. Numpy is widely used in scientific computing, data analysis, and machine learning tasks.

### Features

#### 1.N-dimensional arrays: 
The core functionality of Numpy revolves around its ndarray object, which allows efficient storage and manipulation of arrays. Numpy arrays are homogeneous and can be of any dimensionality, making them ideal for representing vectors, matrices, and tensors.
#### 2.Efficient Computation:
Numpy leverages highly optimized C code under the hood, making it significantly faster than regular Python lists for numerical operations. It provides a wide range of functions for array manipulation, mathematical operations, linear algebra, Fourier transforms, random number generation, and more.
#### 3.Broadcasting:
Numpy's broadcasting feature enables element-wise operations between arrays of different shapes and sizes. It eliminates the need for explicit looping and enhances code readability and performance.
#### 4.Integration with Other Libraries:
Numpy seamlessly integrates with other popular Python libraries, such as Pandas, SciPy, Matplotlib, and scikit-learn. This integration allows for efficient data manipulation, analysis, visualization, and machine learning workflows.
#### 5.Array Indexing and Slicing:
Numpy provides powerful indexing and slicing capabilities to access and modify specific elements or subarrays within an array. These operations can be performed using integer or boolean array indexing, as well as using logical conditions.
#### 6.Linear Algebra Operations:
Numpy offers a comprehensive suite of linear algebra functions, including matrix multiplication, matrix factorization, eigenvalue computation, and solving linear systems of equations. These capabilities are crucial for many scientific and engineering applications.
#### 7.Memory Efficiency: 
Numpy arrays are memory-efficient, as they store data in a contiguous block, reducing overhead and allowing efficient memory access. Numpy also provides features for memory-mapped arrays, which enable reading and writing large arrays stored on disk.

## Installation
You can install the numpy package using the following command.  
              ` pip install numpy `   
              
## Usage
Once installed, you can import Numpy in your Python scripts or interactive sessions using the following import statement:  

` import numpy as np `

## Pandas
Pandas is a powerful and popular open-source data manipulation and analysis library for Python. It provides easy-to-use data structures and data analysis tools, making it an essential tool for data scientists and analysts.

Pandas has two data structures, Series and DataFrame.
### Series and Data Frame
Series is a One dimensional labeled array that hold any data type. It is similar to a column in a spreadsheet or a single column of a database table.On the other hand, a DataFrame is a two-dimensional labeled data structure that resembles a table or a spreadsheet. It consists of multiple columns, each of which can have different data types.

Way's to create the DataFrame. Two ways we can create pandas DataFrame. Using .CSV and .XLSX files. Here's how we can use those to create the DataFrame.

` df = pd.read_csv('filename.csv') `

or 

`df = pd.read_excel('filename.xlsx', 'Sheet1')`

### Missing data in Data frame

There are different ways to handle the missing data in a data frame.

#### fillna
In pandas fillna is used to fill the missing values  ( represented as NaN) in pandas series or data frame with the specific values mentioned or strategies.
         `df = df.fillna(0) `

we can also fill the missing values with the previous value or the next value in a column or row using the below methods.

         `df = df.fillna(method='ffill') `
         `df = df.fillna(method='bfill') `

### dropna
The dropna function in the Pandas library is used to remove missing or null values from a DataFrame. It allows you to drop rows or columns that contain any missing values or only those that have missing values in specific columns.

                    `df = df.dropna() `


### interpolate

The interpolate function in the Pandas library is used to fill missing values in a DataFrame or Series by interpolating between existing values. Interpolation is a technique used to estimate values between known data points based on the pattern or trend in the data.

                    `df = df.interpolate() `
        


## Matplotlib

