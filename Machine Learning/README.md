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
   
2. Suppose you want to save the trained Machine learning model

                                 import joblib
                                 model = ....
                                 joblib.dump(model,"file_path")
3. Suppose you want to load the saved model using joblib:
                                 model = joblib.load("file_path")

##### Using Pickle
1. Install the Pickle library

                                 pip install pickle
   
2. Save the trained ML model using pickle.
                               
                                 import pickle
                                 model= ......
                                 with open ('file_path', 'wb') as f:
                                    pickle.dump(model,f)
3. Load the saved model using pickle
                                 with open ('file_path', 'rb') as f:
                                    model = pickle.load(f)

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
Decision trees are a fundamental and popular machine learning algorithm used for both classification and regression tasks. They are a part of the broader family of supervised learning algorithms, which means they require labeled training data to learn from.
The basic idea behind decision trees is to create a tree-like model that makes a sequence of decisions based on input features until it arrives at a prediction or decision. Each internal node of the tree represents a decision based on a particular feature, and each leaf node represents the final decision or prediction.

                              from sklearn.tree import DecisionTreeClassifier
                              dt = DecisionTreeClassifier()
#### Random forest
Random Forest is a popular ensemble learning algorithm used in machine learning for both classification and regression tasks. It is an extension of the decision tree algorithm, combining multiple decision trees to make more accurate predictions and reduce overfitting. The name "Random Forest" comes from the idea of creating a "forest" of decision trees, where each tree is built using a random subset of the data and features.

                              from sklearn.ensemble import RandomForestClassifier
                              rf = RandomForestClassifier(n_estimators=100)

You can adjust hyperparameters like n_estimators (the number of trees in the forest) and others to optimize the performance of the model for your specific task


#### Support Vector Machine(SVM)
Support Vector Machine (SVM) is a popular and powerful supervised machine learning algorithm used for classification and regression tasks. It is particularly effective in high-dimensional spaces and is widely used in various real-world applications, such as image recognition, text classification, and bioinformatics.
The main idea behind SVM is to find the hyperplane that best separates different classes in the data. In a binary classification setting, this hyperplane is chosen in such a way that it maximizes the margin, which is the distance between the hyperplane and the closest data points (called support vectors) from each class. The larger the margin, the better the generalization performance of the model, as it helps to avoid overfitting.

Key concepts and components of Support Vector Machines:

 1. Hyperplane: In a two-dimensional space (two features), the hyperplane is a simple line that separates the two classes. In higher-dimensional spaces, it becomes a hyperplane.

 2. Support Vectors: These are the data points that lie closest to the hyperplane and influence its position and orientation. These points are used to define the margin and the decision boundary of the SVM.

 3. Margin: The margin is the distance between the hyperplane and the closest support vectors from each class. The goal is to maximize this margin during the training process.

 4. Soft Margin (C-parameter): In real-world datasets, it is not always possible to find a hyperplane that perfectly separates all data points due to noise or overlapping classes. The soft margin allows some misclassifications and introduces a penalty term (C-parameter) for misclassified points.

 5. Kernel Trick: SVM can be extended to handle non-linearly separable data using the kernel trick. It maps the original feature space into a higher-dimensional space, where the data becomes linearly separable. Common kernel functions include polynomial, radial basis function (RBF), and sigmoid kernels.

                              from sklearn.svm import SVC
                              model = SVC()

                              
#### K-Fold Cross validation
K-Fold cross-validation is a widely used technique in machine learning for assessing the performance of a model and reducing overfitting. It involves splitting the dataset into K subsets (or folds) of approximately equal size. The model is then trained and evaluated K times, each time using a different fold as the validation set and the remaining folds as the training set.

Once you have completed K-Fold cross-validation, you can choose the best hyperparameters and model architecture based on the average performance metric. After this, you can retrain the model using the entire dataset and the chosen hyperparameters for deployment.

K-Fold cross-validation is beneficial because it provides a more reliable estimate of a model's performance compared to a single train-test split. It also ensures that the model is tested on different data points, reducing the risk of overfitting to a specific training-validation split.
However, it's essential to keep in mind that K-Fold cross-validation can be computationally expensive, as it involves training and evaluating the model K times. If your dataset is very large or your model is computationally intensive, you might consider using a lower value of K or other cross-validation techniques like stratified K-Fold, leave-one-out cross-validation, or hold-out validation.

                              from sklearn.model_selection import KFold
                              kf = KFold(n_splits = 5)
                              kf.split(data)
                              
KFold is a method to create the cross-validation splits. It splits the data into K folds, where each fold is used as a validation set exactly once while the other K-1 folds are used for training. This is the core function used to implement K-Fold cross-validation.
KFold is not used for directly evaluating the model, but rather it helps you generate the train-test indices for cross-validation.

##### Cross_val_score
cross_val_score is a convenient function in scikit-learn that performs cross-validation and directly returns the evaluation metric(s) for each fold as an array.It combines the process of creating the cross-validation splits (using an internal KFold) and training/evaluating the model for each fold.It automatically handles the process of fitting the model, predicting on the validation set, and calculating the desired evaluation metric(s).cross_val_score makes it easy to get an array of scores for each fold and provides a quick way to compute the average performance metric.

                              from sklearn.model_selection import cross_val_score
                              cross_val_score(model(),X,y,cv=5, scoring='accuracy')

Here CV represents the cross-validation strategy.It can be a number or predefined K-Fold strategy.

##### Stratified K-Fold
Stratified K-Fold cross-validation is a variation of the traditional K-Fold cross-validation that addresses the issue of imbalanced class distributions in the target variable (labels) of the dataset. It ensures that each fold's training and validation sets have approximately the same proportion of samples for each class as the entire dataset.

In standard K-Fold cross-validation, the data is randomly divided into K subsets (folds), and in each iteration, one fold is used as the validation set, and the rest of the folds are used for training. However, this random splitting can lead to some folds having an unequal distribution of classes, especially when the target variable has imbalanced classes.

Stratified K-Fold cross-validation, on the other hand, takes the class distribution into account and creates folds that preserve the relative class frequencies of the target variable. It ensures that each fold is representative of the overall class distribution in the dataset.
By using Stratified K-Fold cross-validation, you can get more reliable estimates of model performance, especially when dealing with imbalanced datasets. It helps ensure that all classes have equal representation during the model evaluation process, reducing the risk of biased performance metrics.

                              from sklearn.model_selection import StratifiedKFold
                              kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
                              scores = cross_val_score(model,X,y,cv=kf,scoring = 'accuracy')


#### K-Means clustering
#### Naive Bayes
#### Hyperparameter tuning
#### Regularization
