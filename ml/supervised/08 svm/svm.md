To find **support vectors**, we need to solve an optimization problem that identifies the hyperplane with the **maximum margin** between classes. Here’s the step-by-step **algorithm** for identifying support vectors using the **Support Vector Machine (SVM)** framework.



### **1. Problem Setup**
We aim to find a hyperplane defined as:
$$
w \cdot x + b = 0
$$
where:
- $w$ is the weight vector (direction of the hyperplane).
- $b$ is the bias term (offset of the hyperplane).
- $x$ is the input vector (features).



### **2. Define the Margin**
The margin is the distance between the hyperplane and the nearest data points from both classes. The goal is to **maximize the margin**, defined as:
$$
\text{Margin} = \frac{2}{\|w\|}
$$
Thus, maximizing the margin is equivalent to minimizing $\|w\|$ (the magnitude of the weight vector), while ensuring that all points are correctly classified.



### **3. Optimization Problem**
The SVM optimization problem can be formulated as:
$$
\text{Minimize: } \frac{1}{2} \|w\|^2
$$
subject to the following constraints:
$$
y_i (w \cdot x_i + b) \geq 1 \quad \text{for all } i
$$
where:
- $y_i$ is the label of the $i$-th data point ($+1$ or $-1$).
- $x_i$ is the feature vector of the $i$-th data point.

The constraints ensure that:
- Points with $y_i = +1$ lie on or above the $w \cdot x + b = +1$ boundary.
- Points with $y_i = -1$ lie on or below the $w \cdot x + b = -1$ boundary.



### **4. Solve Using Lagrange Multipliers**
To solve this constrained optimization problem, we use **Lagrange multipliers**. The Lagrangian is defined as:
$$
\mathcal{L}(w, b, \alpha) = \frac{1}{2} \|w\|^2 - \sum_{i=1}^N \alpha_i \left[ y_i (w \cdot x_i + b) - 1 \right]
$$
where:
- $\alpha_i \geq 0$ are the Lagrange multipliers (one for each training example).



### **5. Dual Formulation**
The optimization problem is reformulated as a dual problem:
$$
\text{Maximize: } \sum_{i=1}^N \alpha_i - \frac{1}{2} \sum_{i=1}^N \sum_{j=1}^N \alpha_i \alpha_j y_i y_j (x_i \cdot x_j)
$$
subject to:
$$
\sum_{i=1}^N \alpha_i y_i = 0, \quad \alpha_i \geq 0
$$



### **6. Identifying Support Vectors**
- Solve the dual optimization problem using methods like **quadratic programming (QP)** to find $\alpha_i$ values.
- Only data points with $\alpha_i > 0$ are **support vectors**. These are the points that lie on the margin boundary or are misclassified (in the case of a soft-margin SVM).

The support vectors are:
$$
x_i \text{ such that } \alpha_i > 0
$$



### **7. Compute the Parameters**
Once the support vectors are identified:
1. **Weight Vector ($w$)**:
   $$
   w = \sum_{i=1}^N \alpha_i y_i x_i
   $$
2. **Bias ($b$)**:
   Using any support vector $x_s$, compute:
   $$
   b = y_s - w \cdot x_s
   $$



### **Example Walkthrough**

#### Training Dataset:
Suppose you have the following data points:
- Class $+1$: $(2, 2), (4, 4)$
- Class $-1$: $(1, 1), (3, 3)$

1. **Formulate the Dual Problem**:
   Maximize:
   $$
   \sum_{i=1}^4 \alpha_i - \frac{1}{2} \sum_{i=1}^4 \sum_{j=1}^4 \alpha_i \alpha_j y_i y_j (x_i \cdot x_j)
   $$
   Subject to:
   $$
   \sum_{i=1}^4 \alpha_i y_i = 0, \quad \alpha_i \geq 0
   $$

2. **Solve for $\alpha_i$**:
   Use a QP solver to find $\alpha_1, \alpha_2, \dots, \alpha_4$.

3. **Identify Support Vectors**:
   Select $x_i$ for which $\alpha_i > 0$. These are your support vectors.

4. **Compute $w$ and $b$**:
   Use the formulas above to compute the weight vector $w$ and bias $b$.

