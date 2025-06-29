The algorithm to train **logistic regression** is based on **maximum likelihood estimation (MLE)**, which typically requires iterative optimization techniques because the likelihood function does not have a closed-form solution. Below is a step-by-step explanation of the training algorithm:

---

### 1. **Logistic Regression Basics**
Logistic regression models the probability of a binary outcome \( y \in \{0, 1\} \) given a set of input features \( X \):
\[
P(y=1 | X) = \sigma(z) = \frac{1}{1 + e^{-z}}, \quad \text{where } z = X \cdot w + b
\]
Here:
- \( w \): Weight vector (coefficients).
- \( b \): Intercept (bias).
- \( \sigma(z) \): Sigmoid function, which maps \( z \) to a range between 0 and 1.

The predicted probability \( P(y=1 | X) \) is:
\[
P(y | X) = \sigma(X \cdot w + b)
\]

---

### 2. **Define the Loss Function**
The training process optimizes the **log-likelihood** of the observed data or minimizes its negative, called the **log-loss (cross-entropy loss)**:

\[
\mathcal{L}(w, b) = - \frac{1}{N} \sum_{i=1}^N \Big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \Big]
\]

where:
- \( \hat{y}_i = \sigma(X_i \cdot w + b) \): Predicted probability for sample \( i \).
- \( y_i \): Actual label for sample \( i \) (\( 0 \) or \( 1 \)).
- \( N \): Number of samples.

This loss is convex, so it can be minimized using optimization techniques.

---

### 3. **Optimization Using Gradient Descent**
To minimize the log-loss, we compute the gradients of \( \mathcal{L}(w, b) \) with respect to \( w \) and \( b \), and update these parameters iteratively.

#### a. **Compute Gradients**
The gradients of the loss with respect to the parameters are:
- For weights \( w \):
  \[
  \frac{\partial \mathcal{L}}{\partial w} = \frac{1}{N} \sum_{i=1}^N \Big[ (\hat{y}_i - y_i) X_i \Big]
  \]
- For the bias \( b \):
  \[
  \frac{\partial \mathcal{L}}{\partial b} = \frac{1}{N} \sum_{i=1}^N \Big[ \hat{y}_i - y_i \Big]
  \]

Here:
- \( \hat{y}_i = \sigma(X_i \cdot w + b) \): Predicted probability.

#### b. **Gradient Descent Update**
The parameters are updated using gradient descent:
\[
w \gets w - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}
\]
\[
b \gets b - \eta \cdot \frac{\partial \mathcal{L}}{\partial b}
\]
where:
- \( \eta \): Learning rate.

---

### 4. **Steps in Training Logistic Regression**
1. **Initialize Parameters**: Start with random or zero values for \( w \) and \( b \).
2. **Iterate Until Convergence**:
   - Compute the predicted probabilities \( \hat{y}_i \) using the sigmoid function.
   - Calculate the gradients \( \frac{\partial \mathcal{L}}{\partial w} \) and \( \frac{\partial \mathcal{L}}{\partial b} \).
   - Update the parameters \( w \) and \( b \) using gradient descent or a variant like stochastic gradient descent (SGD).
3. **Convergence**: Stop the iterations when the change in loss between iterations is below a threshold or when a maximum number of iterations is reached.

---

### 5. **Optimization Variants**
- **Stochastic Gradient Descent (SGD)**: Update the parameters after each data point instead of the entire dataset. Faster but noisier.
- **Mini-batch Gradient Descent**: Update the parameters after a small batch of data points. Balances speed and stability.
- **Advanced Optimizers**: Use optimizers like **Adam**, **RMSProp**, or **L-BFGS**, which adaptively adjust learning rates for faster convergence.

---

### 6. **Regularization**
To prevent overfitting, **regularization** terms can be added to the loss function:
- **L2 Regularization** (Ridge):
  \[
  \mathcal{L}(w, b) = - \frac{1}{N} \sum_{i=1}^N \Big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \Big] + \frac{\lambda}{2} \|w\|^2
  \]
- **L1 Regularization** (Lasso):
  \[
  \mathcal{L}(w, b) = - \frac{1}{N} \sum_{i=1}^N \Big[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \Big] + \lambda \|w\|_1
  \]

---

### Summary of Algorithm:
1. Initialize parameters \( w \) and \( b \).
2. Compute predicted probabilities using the sigmoid function.
3. Calculate gradients of the loss function.
4. Update parameters using gradient descent or a variant.
5. Repeat steps 2–4 until convergence. 

This approach ensures the model learns the optimal parameters \( w \) and \( b \) to minimize the cross-entropy loss.