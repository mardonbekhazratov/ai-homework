### Formula
Let's calculate the derivative of the cost function $F(\beta)$ with respect to $\beta_0$.

The cost function is:
$$
F(\beta) = \frac{1}{n}\sum(y - X\cdot \beta)^2 = \frac{1}{n}(y-X \cdot \beta)^T (y - X\cdot \beta)
$$

Let $\hat{y} = X \cdot \beta$, the predicted values.
The residual vector is :
$$r = y - \hat{y} = y - X\cdot \beta$$
The cost function becomes:
$$ F(\beta) = \frac{1}{n}\sum r_i^2 $$

Then the partial derivative with respect to any $\beta_j$ is:
$$\frac{\partial F}{\partial \beta_j} = -\frac{2}{n}\sum r_i \cdot x_{ij}$$
where:
- $r_i = y_i - \hat{y}_i$ (the residual for the i-th sample),
- $x_{ij} $ is the $j$-thfeature value for the $i$-th sample.

In matric form:
$$\frac{\partial F}{\partial \beta_j} = -\frac{2}{n}\cdot x_j^T \cdot r$$

Great question! Calculating the partial derivatives helps us adjust the coefficients by using them to update the values of \( \beta \) in the direction that minimizes the cost function \( F(\beta) \). This process is called **gradient descent**.

---

### Gradient Descent Formula
The general update rule for gradient descent is:
$$
\beta_j^{\text{new}} = \beta_j^{\text{old}} - \text{step size} \cdot \frac{\partial F}{\partial \beta_j}
$$
Where:
- $ \beta_j^{\text{new}} $ is the updated value of the coefficient,
- $ \beta_j^{\text{old}} $ is the current value of the coefficient,
- $ \text{step size} $ (or learning rate) controls how large each adjustment step is,
- $ \frac{\partial F}{\partial \beta_j} $ is the partial derivative of the cost function with respect to $ \beta_j $, which tells us how $ F(\beta) $ changes as $ \beta_j $ changes.

### Key Generalizations
1. **Gradients**:
   - Instead of manually calculating each partial derivative (\( \beta_0, \beta_1, \beta_2 \)), the script uses matrix multiplication:
     \[
     \text{gradients} = -\frac{2}{n} \cdot X^T \cdot \text{residuals}
     \]
   - \( X^T \) is the transpose of the feature matrix \( X \), allowing the calculation of all gradients at once.

2. **Coefficient Updates**:
   - All coefficients (\( \beta \)) are updated simultaneously in a single line:
     \[
     \beta = \beta - \text{step\_size} \cdot \text{gradients}
     \]


---

### Step-by-Step Process
1. **Calculate Partial Derivatives:** Using the current coefficients, compute \( \frac{\partial F}{\partial \beta_0} \), \( \frac{\partial F}{\partial \beta_1} \), and \( \frac{\partial F}{\partial \beta_2} \) as we just did.

2. **Update Each Coefficient:** Apply the update rule for all coefficients:
   \[
   \beta_0^{\text{new}} = \beta_0^{\text{old}} - \text{step size} \cdot \frac{\partial F}{\partial \beta_0}
   \]
   \[
   \beta_1^{\text{new}} = \beta_1^{\text{old}} - \text{step size} \cdot \frac{\partial F}{\partial \beta_1}
   \]
   \[
   \beta_2^{\text{new}} = \beta_2^{\text{old}} - \text{step size} \cdot \frac{\partial F}{\partial \beta_2}
   \]

3. **Repeat:** Use the updated coefficients to calculate the new partial derivatives and repeat the process until the cost function \( F(\beta) \) converges to a minimum (or until you reach a predefined number of iterations).


---


$n = 5$

$$
X = \begin{pmatrix} 1 & 1 & 3 \\
                    1 & 3 & 4 \\
                    1 & 2 & 6 \\
                    1 & 4 & 1 \\
                    1 & 1 & 7
 \end{pmatrix}
$$

$$
y = \begin{pmatrix} 5 \\ 7 \\ 10 \\5 \\ 6 \end{pmatrix}
$$

$$
\beta = \begin{pmatrix} \beta_0 \\ \beta_1 \\ \beta_2 \end{pmatrix} 
$$

Cost Function:
$$
F(\beta) = \frac{1}{n}\sum\left( y - X \cdot \beta \right) ^ 2 \to min
$$

---

step_size = 0.1
iterations = 1000
Initial random coefficients:
$$
\beta = \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix}
$$

First iteration:
$$
\begin{pmatrix} 1 & 1 & 3 \\
                    1 & 3 & 4 \\
                    1 & 2 & 6 \\
                    1 & 4 & 1 \\
                    1 & 1 & 7
 \end{pmatrix} \cdot 
 \begin{pmatrix} 1 \\ 2 \\ 3 \end{pmatrix} = 
 \begin{pmatrix} 
    1 \cdot 1 + 1 \cdot 2 + 3 \cdot 3 \\
    1 \cdot 1 + 3 \cdot 2 + 4 \cdot 3\\
    1 \cdot 1 + 2 \cdot 2 + 6 \cdot 3\\
    1 \cdot 1 + 4 \cdot 2 + 1 \cdot 3\\
    1 \cdot 1 + 1 \cdot 2 + 7 \cdot 3
 \end{pmatrix} =
 \begin{pmatrix} 12 \\ 29 \\ 23 \\ 12 \\ 24 \end{pmatrix}
$$

$$
MSE(1, 2, 3) = 
\frac{1}{5} \sum\Bigg( \begin{pmatrix} 5 \\ 7 \\ 10 \\5 \\ 6 \end{pmatrix} - \begin{pmatrix} 12 \\ 29 \\ 23 \\ 12 \\ 24 \end{pmatrix} \Bigg) ^ 2 =
\frac{1}{5} \sum \begin{pmatrix} 49 \\ 144 \\ 169 \\ 49 \\ 324 \end{pmatrix} = \frac{1}{5} (49 + 144 + 169+ 49 + 324) = \frac{735}{5} = 147
$$

$$
\frac{\partial F(\beta)}{\partial \beta} = -\frac{2}{n} \cdot X^T \cdot r = 
-\frac{2}{5} \begin{pmatrix}
    1 & 1 & 1 & 1 & 1 \\
    1 & 3 & 2 & 4 & 1 \\
    3 & 4 & 6 & 1 & 7
\end{pmatrix} \cdot \begin{pmatrix} -7 \\ -12 \\ -13 \\ -7 \\ -18 \end{pmatrix} = \begin{pmatrix} 22.8 \\ 46.0 \\ 112.0 \end{pmatrix}
$$

$$
\beta = \beta - \text{stepsize} \cdot \frac{\partial F(\beta)}{\partial \beta}
$$

$$
\beta = \begin{pmatrix}1 \\ 2 \\ 3 \end{pmatrix} - 0.1 \cdot \begin{pmatrix} 22.8 \\ 46.0 \\ 112.0 \end{pmatrix} = \begin{pmatrix} -1.28 \\ -2.6 \\ -8.2 \end{pmatrix}
$$

Next iterations:
**REPEAT THE SAME ALGORITHM**

---
