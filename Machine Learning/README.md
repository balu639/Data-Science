## Machine Learning
Machine learning is a subfield of artificial intelligence (AI) that focuses on developing algorithms and techniques that enable computers to learn and improve their performance on a specific task without being explicitly programmed. The core idea behind machine learning is to allow machines to learn from data, identify patterns, and make predictions or decisions based on that learned knowledge.

### Types of Machine learning

#### Supervised Learning
In this approach, the algorithm is trained on a labeled dataset, where the input data and their corresponding correct outputs are provided. The algorithm learns to map inputs to outputs by generalizing patterns from the training data. Common tasks in supervised learning include classification (assigning inputs to predefined categories) and regression (predicting continuous values).

Algorithms are 
 * Linear Regression
 * Logistic Regression
 * Decision Trees
 * Random Forest
 * Support Vector Machine(SVM)
 * KNN(K-Nearest Neighbours)

#### Unsupervised Learning
In unsupervised learning, the algorithm is trained on an unlabeled dataset, and it has to find patterns and structures within the data on its own. Clustering and dimensionality reduction are examples of unsupervised learning tasks.

Algorithms are
 * K-Means Clustering
 * Principal Component Analysis(PCA)

#### Semi-supervised Learning
This is a combination of supervised and unsupervised learning, where the algorithm is trained on a partially labeled dataset. It uses both labeled and unlabeled data to improve its performance.

#### Reinforcement Learning
In this paradigm, an agent learns to interact with an environment to achieve a goal. The agent receives feedback in the form of rewards or penalties based on its actions, and it learns to take actions that maximize cumulative reward over time.


### Machine Learning Algorithms

#### Linear regression
Linear regression is a fundamental and widely used statistical technique for modeling the relationship between a dependent variable (also known as the target or response variable) and one or more independent variables (also known as predictors or features). It is a supervised learning algorithm commonly applied in various fields, including economics, finance, social sciences, and machine learning.
Linear regression is commonly used for regression tasks.

##### simple linear regression
Let's start with the simplest form of linear regression, known as simple linear regression. In this case, we have a single independent variable (X) and a single dependent variable (Y). The relationship between X and Y can be approximated by a straight line equation:
                            
                            Y = b0 + b1 * X

Here, b0 represents the intercept (the value of Y when X is 0), and b1 is the slope of the line (the change in Y corresponding to a unit change in X). The goal of simple linear regression is to find the best-fitting line that minimizes the distance between the actual data points and the predicted values along this line.

##### Multi-variate linear regression
When dealing with multiple independent variables, we use multi-variate linear regression. The equation becomes:

                          Y = b0 + b1 * X1 + b2 * X2 + ... + bn * Xn
##### Usage 

                          from sklearn.linear_model import LinearRegression
                          lr = LinearRegression()
#### Logistic regression
Logistic regression is a popular statistical method used for binary classification tasks, where the goal is to predict a binary outcome or dependent variable based on one or more independent variables (also known as features or predictors). It is a type of generalized linear model (GLM) and is widely used in various fields, such as machine learning, statistics, and data analysis.

The logistic regression model uses the logistic function (also known as the sigmoid function) to map the input features to a probability value between 0 and 1. The formula for the logistic function is:

                         sig(z) = 1/1+e ^ -z

where z is the linear combination of the input features and their corresponding coefficients:
                        z = b0 + b1 * X1 + b2 * X2 + ... + bn * Xn

                        
#### Gradient descent
Gradient descent is an optimization algorithm commonly used in machine learning and deep learning to find the minimum of a function. It is a first-order optimization algorithm that relies on the gradient of the function, which indicates the direction of the steepest increase in the function's value. The goal of gradient descent is to iteratively update the parameters of a model in the direction opposite to the gradient, with the aim of minimizing the function's value.

Here's a basic explanation of how gradient descent works:
 1. Define a Loss Function: In machine learning, we typically have a model that makes predictions, and we want to find the best set of parameters (weights and biases) for the model. To do this, we first define a loss function that quantifies how far off our predictions are from the actual targets. The objective of gradient descent is to minimize this loss function.
 2. Initialization: We start with some initial values for the model's parameters. These can be random or based on prior knowledge.
 3. Calculate Gradient: The gradient of the loss function with respect to each parameter is computed. The gradient is a vector that points in the direction of the steepest increase in the loss function.
 4. Update Parameters: The parameters are updated by moving in the opposite direction of the gradient. This is done by subtracting a fraction of the gradient from the current parameter values. The fraction is determined by a parameter called the learning rate, which controls the step size of each update. A smaller learning rate makes the algorithm more stable but may require more iterations to converge, while a larger learning rate can lead to faster convergence but may risk overshooting the minimum.
 5. Repeat: Steps 3 and 4 are repeated for a specified number of iterations or until the convergence criteria are met. Convergence is usually defined based on a predefined tolerance level or when the change in the loss function becomes negligible.
 6. Convergence: Ideally, the algorithm will converge to a minimum point of the loss function, representing the best parameter values for the model given the training data.

There are different variants of gradient descent, such as Stochastic Gradient Descent (SGD), Mini-batch Gradient Descent, and Batch Gradient Descent. These variants differ in how they use data to compute gradients and update parameters. SGD uses a single data point at a time, Mini-batch Gradient Descent uses a small subset of data points, and Batch Gradient Descent uses the entire dataset in each iteration.


#### Saving a Model
Machine learning models can be saved using Joblib or pickle libraries in python.Both joblib and pickle are libraries that allow you to serialize Python objects, including ML models, so that they can be saved to disk and later loaded and used for predictions.
##### Using Joblib
1. Install the joblib library

                                 pip install joblib
   
3. Suppose you want to save the trained Machine learning model

                                 import joblib
                                 model = ....
                                 joblib.dump(model,"file_path")

##### Using Pickle
1. Install the Pickle library

                                 pip install pickle
   
3. Save the trained ML model using pickle.
                               
                                 import pickle
                                 model= ......
                                 with open ('model_pickle', 'wb') as f:
                                    pickle.dump(model,f)

Both joblib and pickle are useful for saving and loading ML models, but joblib is often preferred for large numerical arrays typically encountered in ML models, as it is more efficient in handling these cases. However, it's worth noting that pickle is a standard Python module and can be used for a broader range of objects beyond just ML models.
   
#### One hot encoding
One-hot encoding is a popular technique used in machine learning and natural language processing to convert categorical data into a numerical representation. It is particularly useful when dealing with categorical variables that have no inherent order or numerical value.

In one-hot encoding, each unique category in the categorical variable is represented as a binary vector, where all elements are zero except for the position that corresponds to the category's index. For example, if you have a categorical variable with three possible values: "red," "green," and "blue," one-hot encoding would represent them as follows:

 * "red" -> [1, 0, 0]
 * "green" -> [0, 1, 0]
 * "blue" -> [0, 0, 1]

As you can see, only one element is 1, while the rest are 0s. This way, each category is represented as a distinct vector, and the model can easily identify the presence of a particular category in the data.

In Python, you can use the pandas library to perform one-hot encoding on categorical data easily. The pandas library provides the get_dummies() function, which converts categorical variables into one-hot encoded columns. Here's how you can use it:

                                import pandas as pd
                                dummies = pd.get_dummies(df.column)

#### Train and Test split
In machine learning, the process of splitting a dataset into separate training and testing sets is crucial for model development and evaluation. The primary purpose of this split is to allow the model to learn patterns from the training data and then assess its performance on unseen data (testing data) to estimate how well it generalizes to new examples.

Once the data is preprocessed, split it into two subsets: the training set and the testing set. The training set will be used to train the machine learning model, while the testing set will be used to evaluate its performance.

Commonly, the data is split into a ratio like 70-30, 80-20, or 90-10, depending on the size of the dataset. The training set contains a larger portion of the data to ensure the model has enough examples to learn from, while the testing set remains smaller to provide a reliable evaluation.

In Python, you can use libraries like scikit-learn to perform the train-test split:

                              from sklearn.model_selection import train_test_split
                              X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2, random_state=42)

random_state is a parameter that is used to control the random number generator's seed value. The seed value is an initial value that determines the sequence of random numbers generated by the algorithm. Setting a specific random_state ensures reproducibility, meaning that if you run the same code with the same random_state value, you will get the same split each time.

#### Decision Trees
#### Random forest
#### Support Vector Machine(SVM)
#### K-Fold Cross validation
#### K-Means clustering
#### Naive Bayes
#### Hyperparameter tuning
#### Regularization
