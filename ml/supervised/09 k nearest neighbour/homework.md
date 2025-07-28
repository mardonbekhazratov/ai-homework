### **K-Nearest Neighbors: Predicting Customer Churn**

#### Objective:
Your task is to build a **K-Nearest Neighbors (KNN)** classifier to predict whether a customer will churn or stay based on their usage patterns and demographic data. This will help you understand how KNN works for classification and explore its strengths and limitations.

---

#### Dataset:
Use the **Customer Churn Dataset**, available from:
- [Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)  
- Or other sources such as open repositories.

Alternatively, you can create a simulated churn dataset using Python's libraries like `sklearn.datasets.make_classification`.

---

#### Dataset Overview:
Key features:
- **Demographic Information**: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
- **Account Details**: `tenure`, `Contract`, `MonthlyCharges`, `TotalCharges`
- **Services Used**: `PhoneService`, `MultipleLines`, `InternetService`, etc.

Target variable:
- **Churn**: Binary column indicating whether a customer churned (`Yes` or `No`).

---

#### Steps to Complete:

1. **Load the Dataset**
   - Import the dataset using `pandas`.
   - Display the first few rows to understand the structure.
   - Identify categorical and numerical columns.

2. **Data Exploration**
   - Check the distribution of the target variable (`Churn`).
   - Explore correlations between numerical features (e.g., `tenure`, `MonthlyCharges`, `TotalCharges`) and `Churn`.
   - Use group-by operations to analyze churn rates across different categories (e.g., `Contract`, `InternetService`).

3. **Data Cleaning**
   - Handle missing values (e.g., `TotalCharges` might have nulls).
   - Encode categorical features using one-hot encoding or label encoding.
   - Normalize/scale numerical features (important for KNN).

4. **Data Splitting**
   - Split the dataset into training (80%) and testing (20%) sets using `train_test_split`.

5. **Build and Evaluate KNN Model**
   - Train a KNN classifier using `KNeighborsClassifier` from `sklearn.neighbors`.
   - Use the default value of `k` (number of neighbors).
   - Evaluate the model on the test set using:
     - **Accuracy**
     - **Confusion Matrix**
     - **Classification Report** (precision, recall, F1-score)

6. **Optimize the K Value**
   - Experiment with different values of `k` (e.g., 1, 3, 5, 7, 10).
   - Plot a graph of `k` vs. accuracy to identify the optimal value.
   - Use cross-validation to confirm the chosen `k`.

7. **Compare Distance Metrics**
   - Train and evaluate the KNN model with different distance metrics:
     - **Euclidean distance** (default)
     - **Manhattan distance**
     - **Minkowski distance**
   - Analyze which metric works best for this dataset.

8. **Insights**
   - Discuss the overall performance of the KNN model.
   - Highlight the importance of scaling and choosing an optimal `k`.
   - Reflect on potential challenges when using KNN for larger datasets.

---

#### Bonus Challenges (Optional):

1. **Feature Selection**
   - Use feature importance techniques (e.g., correlation analysis or mutual information) to identify the most relevant features.
   - Train the KNN model using only these features and compare its performance to the full model.

2. **Customer Profiling**
   - Create customer profiles (e.g., high risk, low risk) based on their churn probabilities.

3. **Comparison with Other Models**
   - Compare KNN's performance to other models like:
     - **Logistic Regression**
     - **Random Forest**

4. **Advanced Visualization**
   - Use **t-SNE** or **PCA** to reduce the feature space to 2D and visualize the clusters of churned and non-churned customers.

---

#### Deliverables:
- A Python script or Jupyter Notebook containing:
  - Data exploration, cleaning, and preprocessing.
  - Implementation of KNN with optimized `k` and distance metrics.
  - Visualizations (e.g., churn distributions, `k` vs. accuracy plot).
- A brief report discussing:
  - Key insights from the dataset.
  - How the model performs with different `k` values and distance metrics.
  - Challenges encountered during data preparation or modeling.

---

#### Useful Hints:
- Use `seaborn` for visualizations like heatmaps and bar plots.
- Normalize numerical features using `StandardScaler` or `MinMaxScaler` to ensure fair distance calculations.
- For imbalanced datasets, consider using `stratify` in `train_test_split`.

