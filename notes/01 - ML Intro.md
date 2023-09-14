## 1. Introduction to Machine Learning

The concept of ML is depicted with an example of predicting the price of a car. The ML model learns from data, represented as some **features** such as year, mileage, among others, and the **target** variable, in this case, the car's price, by extracting patterns from the data.

Then, the model is given new data (**without** the target) about cars and predicts their price (target).

In summary, ML is a process of **extracting patterns from data**, which is of two types:
- features (information about the object) and
- target (property to predict for unseen objects).

Therefore, new feature values are presented to the model, and it makes **predictions** from the learned patterns.

---
## 2. ML vs Rule-Based Systems

ML is a _**paradigm shift**_ compared to traditional programming. Traditional programming follows this structure:

`data + code => outcome`

But ML changes this equation and becomes like this:

`data + outcome => model`


The differences between ML and Rule-Based systems can be explained with the example of a **spam filter**.

Traditional Rule-Based systems are based on a set of **characteristics** (keywords, email length, etc.) that identify an email as spam or not. As spam emails keep changing over time the system needs to be upgraded making the process untractable due to the complexity of code maintenance as the system grows.

ML can be used to solve this problem with the following steps:

##### 2.1. Get data
Emails from the user's spam folder and inbox gives examples of spam and non-spam.

##### 2.2. Define and calculate features
Rules/characteristics from rule-based systems can be used as a starting point to define features for the ML model. The value of the target variable for each email can be defined based on where the email was obtained from (spam folder or inbox).

Each email can be encoded (converted) to the values of it's features and target.

##### 2.3. Train and use the model
A machine learning algorithm can then be applied to the encoded emails to build a model that can predict whether a new email is spam or not spam. The **predictions are probabilities**, and to make a decision it is necessary to define a threshold to classify emails as spam or not spam.

---
## 3. Supervised Machine Learning
In Supervised Machine Learning (SML) there are always labels associated with certain features. The model is trained, and then it can make predictions on new features. In this way, the model is taught by certain features and targets.

- **Feature matrix (X):** made of observations or objects (rows) and features (columns).
- **Target variable (y):** a vector with the target information we want to predict. For each row of X there's a value in y.

The model can be represented as a function **g** that takes the X matrix as a parameter and tries to predict values as close as possible to y targets. The obtention of the g function is what it is called **training**.

##### Types of SML problems

- **Regression:** the output is a number (car's price)
- **Classification:** the output is a category (spam example).
    - **Binary:** there are two categories.
    - **Multiclass problems:** there are more than two categories.
- **Ranking:** the output is the big scores associated with certain items. It is applied in recommender systems.

In summary, SML is about teaching the model by showing different examples, and the goal is to come up with a function that takes the feature matrix as a parameter and makes predictions as close as possible to the y targets.

---
## 4. CRISP-DM

**CRISP-DM** (_cross-industry standard process for data mining_) is a methodology for organizing ML projects. Even though it was created in the 90's, it's still relevant today and describes the steps to carry out a successful ML project.

![[01_d01.png]]

1. **Business understanding**
    - Analyze the problem. How serious is it and to what extent? Is it just one user complaining or a company-wide issue?
    - Will Machine Learning help? Can the problem be solved with easier methods?
    - Define a **goal** to achieve. The goal must be **measurable**.
        - For the spam example, the goal could be _reduce the amount of spam messages_, or perhaps _reduce the amount of complaints about spam_.
        - Our measue will be to _reduce the amount of spam by 50%_.
2. **Data undestanding**
    - Once you've decided on using ML, analyze and identify available data sources and decide if we need to get more data.
        - Spam example: do we have a _report spam_ button? Is the data behind this button good enough? Is the button reliable? Do we track spam correctly? Is our dataset large enough? Do we need to get more data?
    - Understanding the data may give us new insights into the problem and influence the goal. We may go back to the Business Understanding step and adjust it accordingly.
3. **Data preparation**
    - Transform the data so it can be put into a ML algorithm.
        - Clean the data (remove noise)
        - Build the pipelines (raw data -> transformations -> clean data)
        - Convert into tabular form
4. **Modeling**
    - Training the models. The actual Machine Learning happens here.
        - Try different models
        - Select the best one
    - Which model to choose?
        - Logistic regression
        - Decission tree
        - Neural network
        - Many others!
    - Sometimes we may have to go back to the Data Preparation step to add new features or fix data issues.
5. **Evaluation**
    - Measure how well the model solves the business problem.
    - Is the model good enough?
        - Have we reached our goal?
        - Do our metrics improve?
    - Do a retrospective:
        - Was the goal achievable?
        - Did we solve/measure the right thing?
    - After the retrospective, we may decide to:
        - Go back to the Business Understanding step and adjust the goal
        - Roll the model to more/all users
        - Stop working on the project (!)
6. **Evaluation + Deployment**
    - Often the 2 steps happen together:
        - Online evaluation: evaluation of live users
        - It means that we deploy first and then evaluate it on a small percentage of users
    - This is where modern practices differ slightly from the original CRISP-DM methodology.
7. **Deployment**
    - Roll the model to all users
    - Proper monitoring
    - Ensuring quality and maintainability
    - Essentially, this is the "engineering" step.
8. **Iterate!**
    - ML projects require many iterations! This also differs from the original CRISP-DM
    - After Deployment, we may go back once again to Business Understanding and wonder whether the project can be improved upon

Additional iteration guidelines:

1. Start simple
2. Learn from feedback
3. Improve

**Iteration:**
- Start simple
- Learn from the feedback
- Improve

---
## 5. Model Selection Process
##### Which model to choose?
- Logistic regression
- Decision tree
- Neural Network
- Or many others

The validation dataset is not used in training. There are feature matrices and y vectors for both training and validation datasets. The model is fitted with training data, and it is used to predict the y values of the validation feature matrix. Then, the predicted y values (probabilities) are compared with the actual y values.

**Multiple comparisons problem (MCP):** just by chance one model can be lucky and obtain good predictions because all of them are probabilistic.

The test set can help to avoid the MCP. Obtaining the best model is done with the training and validation datasets, while the test dataset is used for assuring that the proposed best model is the best.

1. Split datasets in training, validation, and test. E.g. 60%, 20% and 20% respectively
2. Train the models
3. Evaluate the models
4. Select the best model
5. Apply the best model to the test dataset
6. Compare the performance metrics of validation and test

---
## 6. Numpy

[Numpy Cheatsheet](https://www.datacamp.com/cheat-sheet/numpy-cheat-sheet-data-analysis-in-python)

---

## 7. Linear Algebra Refresher

#### 7.1 Vector operations
~~~~python
u = np.array([2, 7, 5, 6])
v = np.array([3, 4, 8, 6])

# addition 
u + v

# subtraction 
u - v

# scalar multiplication 
2 * v
~~~~
#### 7.2 Multiplication

#####  7.2.1 Vector-vector multiplication

~~~~python
def vector_vector_multiplication(u, v):
    assert u.shape[0] == v.shape[0]
    
    n = u.shape[0]
    
    result = 0.0

    for i in range(n):
        result = result + u[i] * v[i]
    
    return result
~~~~

#####  7.2.2 Matrix-vector multiplication

~~~~python
def matrix_vector_multiplication(U, v):
    assert U.shape[1] == v.shape[0]
    
    num_rows = U.shape[0]
    
    result = np.zeros(num_rows)
    
    for i in range(num_rows):
        result[i] = vector_vector_multiplication(U[i], v)
    
    return result
~~~~

#####  7.2.3 Matrix-matrix multiplication

~~~~python
def matrix_matrix_multiplication(U, V):
    assert U.shape[1] == V.shape[0]
    
    num_rows = U.shape[0]
    num_cols = V.shape[1]
    
    result = np.zeros((num_rows, num_cols))
    
    for i in range(num_cols):
        vi = V[:, i]
        Uvi = matrix_vector_multiplication(U, vi)
        result[:, i] = Uvi
    
    return result
~~~~
##### 7.3 Identity matrix

~~~~python
I = np.eye(3)
~~~~
##### 7.4 Inverse
~~~~python
V = np.array([
    [1, 1, 2],
    [0, 0.5, 1], 
    [0, 2, 1],
])
inv = np.linalg.inv(V)
~~~~

---
## 8. Pandas
[Pandas CheatSheet](https://www.datacamp.com/cheat-sheet/pandas-cheat-sheet-for-data-science-in-python)