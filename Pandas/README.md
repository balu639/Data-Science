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

            df = df.fillna(method='ffill') 
            df = df.fillna(method='bfill')

#### dropna
The dropna function in the Pandas library is used to remove missing or null values from a DataFrame. It allows you to drop rows or columns that contain any missing values or only those that have missing values in specific columns.

                     df = df.dropna()


#### interpolate

The interpolate function in the Pandas library is used to fill missing values in a DataFrame or Series by interpolating between existing values. Interpolation is a technique used to estimate values between known data points based on the pattern or trend in the data.

                     df = df.interpolate()
        

