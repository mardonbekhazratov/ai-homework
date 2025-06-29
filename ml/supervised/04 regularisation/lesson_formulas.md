### Calculating Closed Form solution for Ridge Regression

$$
\mathcal{J}(\theta) = \text{MSE}(\theta) + \frac{\alpha}{N} \sum_{i=1}^{m} \theta_i^2
$$

$$
R(\theta) = \frac{\alpha}{N} \sum_{i=1}^{m} \theta_i^2
$$


$$
\frac{\partial R(\theta)}{\partial \theta}
$$



$$
R(\theta) = \frac{\alpha}{N} \sum_{i=1}^m \theta_i^2
$$

This sum of squares can be written in **vector notation** as:

$$
R(\theta) = \frac{\alpha}{m} \cdot \theta^T \theta
$$

We’ll differentiate:

$$
\frac{\partial}{\partial \theta} \left( \theta^T \theta \right)
$$

Let’s write out the dot product:

$$
\theta^T \theta = \sum_{i=1}^n \theta_i^2
$$

So, $f(\theta) = \theta^T \theta$ is a **scalar** function.

We are differentiating this scalar with respect to the **vector** $\theta = [\theta_1, \theta_2, \dots, \theta_m]^T$, which means we are computing the **gradient**:

$$
\frac{\partial}{\partial \theta}(\theta^T \theta) = \begin{bmatrix}\frac{\partial}{\partial \theta_1} \sum_{i=1}^m \theta_i^2 \\ \frac{\partial}{\partial \theta_2} \sum_{i=1}^m \theta_i^2 \\ \vdots \\ \frac{\partial}{\partial \theta_m} \sum_{i=1}^m \theta_i^2 \end{bmatrix} = \begin{bmatrix} 2\theta_1 \\ 2\theta_2 \\ \vdots \\ 2\theta_m \end{bmatrix} = 2\theta
$$

Each component is just the partial derivative of $\theta_i^2$, which is $2\theta_i$.


$$
\frac{\partial}{\partial \theta} \left( \theta^T \theta \right) = 2\theta
$$

So:

$$
\frac{\partial R(\theta)}{\partial \theta} = \frac{\alpha}{N} \cdot \frac{\partial}{\partial \theta} (\theta^T \theta) = \frac{\alpha}{N} \cdot 2\theta = \frac{2\alpha}{N} \theta
$$

Derivative of regularization term:

$$
\boxed{\frac{\partial R(\theta)}{\partial \theta} = \frac{2\alpha}{N} \theta}
$$


#### Final derivative:

$$
\boxed{
\frac{\partial \mathcal{J}(\theta)}{\partial \theta}
= \frac{2}{N} X^T(X\theta - y) + \frac{2\alpha}{N} \theta
}
$$


---

### Closed-Form solution

$$
\frac{\partial \mathcal{J}(\theta)}{\partial \theta}
= \frac{2}{N} X^T(X\theta - y) + \frac{2\alpha}{N} \theta = 0
$$


$$
\frac{2}{N} X^T(X\theta - y) + \frac{2\alpha}{N} \theta = 0
$$

$$
X^T(X\theta - y) + \alpha \theta = 0
$$

$$
X^T X \theta - X^T y + \alpha \theta = 0
$$

Group the $\theta$ terms:

$$
(X^T X + \alpha I)\theta = X^T y
$$

#### Final Solution:

$$
\boxed{
\theta = (X^T X + \alpha I)^{-1} X^T y
}
$$

This is the **closed-form solution** for **ridge regression**, where $\alpha$ is the regularization strength.

---








