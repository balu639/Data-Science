## Matplotlib

Matplotlib is a popular plotting library for the Python programming language. It provides a wide range of functionalities for creating various types of plots and visualizations. Matplotlib is widely used in data analysis, scientific computing, and data visualization tasks.

### usage

                 import matplotlib.pyplot as plt 
                 %matplotlib inline 
                 plt.xlabel() 
                 plt.ylabel() 
                 plt.title() 
                 plt.plot(x,y, color = '' , linestyle='', linewidth='')

### Axes labels, legend, grid
In the Matplotlib library, you can customize various aspects of your plot, including axes labels, legends, and gridlines.

#### Axes labels
X-axis Label: You can set the label for the x-axis using the xlabel() function. For example: plt.xlabel('X-axis label').
Y-axis Label: You can set the label for the y-axis using the ylabel() function. For example: plt.ylabel('Y-axis label').

#### legend
To create a legend for your plot, you need to assign labels to the elements you want to appear in the legend and then call the legend() function.

                    plt.plot(x1, y1, label='Line 1') 
                    plt.plot(x2, y2, label='Line 2')
                    plt.legend()

#### grid
To display major gridlines on both the x and y axes, you can use the grid() function without any arguments. For example:  ` plt.grid() `

You can customize the gridlines further by passing arguments to the grid() function. For example:

                     plt.grid(color='gray', linestyle='--', linewidth=0.5)


### Bar chart
A bar chart is a graphical representation that uses rectangular bars to represent data. Each bar's length corresponds to the data's value it represents. Bar charts are commonly used to compare categorical data or to show the distribution of different categories.

In Matplotlib, you can create bar charts using the `bar()` function. You should use a bar chart in Matplotlib when you want to visualize discrete data points and compare their values. It is suitable for displaying data that can be organized into distinct categories, such as comparing sales figures for different products, population distribution across cities, or exam scores for different subjects.

### Pie chart

A pie chart is a circular graph divided into slices, where each slice represents a proportion of the whole data set. The size of each slice is proportional to the data it represents, showcasing the percentage or fraction of the whole.

In Matplotlib, you can create a pie chart using the `pie()` function. You should use a pie chart when you want to visualize the composition of a whole and its individual parts. It is suitable for displaying data with clear categories and their corresponding proportions, such as the percentage distribution of different expenses in a budget, the market share of various products, or the distribution of votes in an election.


### Histogram
A histogram is a graphical representation that organizes numerical data into bins or intervals. It displays the frequency or count of data points falling within each bin on the y-axis, and the bins themselves on the x-axis.

In Matplotlib, you can create a histogram using the `hist()` function. You should use a histogram when you want to visualize the distribution of continuous or numerical data and understand its frequency distribution, shape, and range. It is suitable for displaying data such as exam scores of students, temperatures recorded over time, or the number of customers in different age groups.

To summarize, use a histogram in Matplotlib to explore the distribution of continuous data and observe patterns in its frequency or count within specific intervals (bins).

