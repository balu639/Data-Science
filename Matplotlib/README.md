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
