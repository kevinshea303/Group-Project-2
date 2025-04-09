# Group-Project-2
---
## GROUP 1 PROJECT 2 TEAM MEMBERS
 Project 2 presented on April 10, 2025 by Group 1 Team Members

 *Michael Hnidash*  
 *Alexander Lin*  
 *Nan Shi*   
 *Kevin Rivera*  
 *Sofia Frrokaj*  

### **Project Overview**

This project involves the analysis and modeling of a dataset (`data.csv`) to predict student outcomes, specifically whether a student will **drop out**, remain **enrolled**, or **graduate**. The analysis includes data preprocessing, feature engineering, model training, hyperparameter optimization, and evaluation of two machine learning models: **Random Forest Classifier** and **Logistic Regression**. The project also explores feature importance and the impact of feature selection on model performance.

---

### **Data Preprocessing and Cleaning**

1. **Loading the Dataset**:
   - The dataset was loaded from the file `data.csv` using the `pandas` library.

2. **Renaming Columns**:
   - The column `Daytime/evening attendance ` was renamed to `Daytime/evening attendance` to remove trailing spaces.

3. **Handling Missing Values**:
   - For **numerical columns**, missing values were imputed with the **mean** of the respective column.
   - For **categorical columns**, missing values were imputed with the **mode** of the respective column.

4. **Encoding Categorical Variables**:
   - The target variable (`Target`) was encoded using `LabelEncoder` to convert class labels into numerical values.
   - Categorical features were converted into dummy variables using one-hot encoding (`pd.get_dummies`), ensuring no missing values in the resulting dataset.

5. **Feature Scaling**:
   - The features were scaled using `StandardScaler` to standardize the data, ensuring that all features have a mean of 0 and a standard deviation of 1. This step is crucial for models sensitive to feature magnitudes, such as Logistic Regression.

---

### **Exploratory Data Analysis (EDA)**

- **Target Variable Distribution**:
  - The target variable (`Target`) represents three classes:
    - **0**: Dropout
    - **1**: Enrolled
    - **2**: Graduate
  - The distribution of these classes was analyzed to identify potential class imbalances.

- **Feature Importance**:
  - Feature importance was calculated using the Random Forest model, providing insights into which features contribute most to the prediction of student outcomes.

---

### **Modeling and Hyperparameter Optimization**

#### **1. Random Forest Classifier**

- **Hyperparameter Tuning**:
  - A hyperparameter grid was defined for the Random Forest model, including parameters such as:
    - `n_estimators`: Number of trees in the forest (e.g., 100, 200, 300).
    - `max_depth`: Maximum depth of the trees (e.g., 5, 10, 15, None).
    - `min_samples_split`: Minimum number of samples required to split an internal node (e.g., 2, 5, 10).
    - `min_samples_leaf`: Minimum number of samples required to be at a leaf node (e.g., 1, 2, 4).
    - `max_features`: Number of features to consider when looking for the best split (e.g., 'sqrt', 'log2').
  - Hyperparameter tuning was performed using `RandomizedSearchCV` with 3-fold cross-validation and 10 iterations.

- **Model Evaluation**:
  - The best model was evaluated on the test set, and metrics such as accuracy, precision, recall, and F1-score were calculated.
  - Feature importance was visualized using a bar plot, highlighting the most influential features.

- **Feature Selection**:
  - Features with importance values above a specified threshold (e.g., 0.01) were selected.
  - A new Random Forest model was trained using only the selected features, and its performance was compared to the original model.

#### **2. Logistic Regression**

- **Hyperparameter Tuning**:
  - A hyperparameter grid was defined for Logistic Regression, including parameters such as:
    - `penalty`: Regularization type (e.g., 'l1', 'l2').
    - `C`: Inverse of regularization strength, tested over a logarithmic scale (e.g., `np.logspace(-4, 4, 20)`).
    - `solver`: Optimization algorithm (e.g., 'liblinear', suitable for L1 and L2 regularization).
  - Hyperparameter tuning was performed using `RandomizedSearchCV` with 3-fold cross-validation and 10 iterations.

- **Model Evaluation**:
  - The best Logistic Regression model was evaluated on the test set, and metrics such as accuracy, precision, recall, and F1-score were calculated.

---

### **Results and Analysis**

#### **1. Random Forest Classifier**

- **Best Parameters**:
  - The optimal hyperparameters identified through `RandomizedSearchCV` were:
    - `n_estimators`: [Optimal value]
    - `max_depth`: [Optimal value]
    - `min_samples_split`: [Optimal value]
    - `min_samples_leaf`: [Optimal value]
    - `max_features`: [Optimal value]

- **Performance**:
  - Test Accuracy: [Value]
  - Classification Report:
    - Precision, Recall, and F1-score for each class (Dropout, Enrolled, Graduate) were reported.
    - The model performed best at identifying **Graduate** students, with high recall and F1-score.
    - The model struggled with **Enrolled** students, showing low recall and precision.

- **Feature Importance**:
  - The most important features contributing to the model's predictions were identified and visualized.
  - Selected features were used to train a new model, which achieved comparable or improved performance.

#### **2. Logistic Regression**

- **Best Parameters**:
  - The optimal hyperparameters identified through `RandomizedSearchCV` were:
    - `penalty`: [Optimal value]
    - `C`: [Optimal value]
    - `solver`: [Optimal value]

- **Performance**:
  - Test Accuracy: [Value]
  - Classification Report:
    - Precision, Recall, and F1-score for each class (Dropout, Enrolled, Graduate) were reported.
    - Similar to the Random Forest model, Logistic Regression performed best at identifying **Graduate** students but struggled with **Enrolled** students.

---

### **Comparison of Models**

- **Accuracy**:
  - Random Forest: [Value]
  - Logistic Regression: [Value]

- **Strengths**:
  - Random Forest demonstrated higher flexibility and better handling of complex relationships in the data.
  - Logistic Regression provided a simpler and more interpretable model.

- **Weaknesses**:
  - Both models struggled with the **Enrolled** class, likely due to class imbalance or overlapping feature distributions.

---

### **Conclusion and Recommendations**

1. **Model Selection**:
   - The Random Forest model is recommended for deployment due to its superior performance and ability to handle feature importance analysis.

2. **Feature Engineering**:
   - Further exploration of feature engineering techniques, such as interaction terms or polynomial features, may improve model performance.

3. **Class Imbalance**:
   - Addressing class imbalance through techniques such as oversampling, undersampling, or class-weight adjustments may improve the model's ability to predict the **Enrolled** class.

4. **Future Work**:
   - Explore additional machine learning models, such as Gradient Boosting or Neural Networks, to compare performance.
   - Conduct a deeper analysis of misclassified instances to identify potential improvements in data preprocessing or feature selection.

---

### **Documentation**

1. **Data Loading**:
   - The dataset was loaded from `data.csv`.

2. **Data Cleaning**:
   - Missing values were imputed using the mean for numerical columns and the mode for categorical columns.

3. **Encoding**:
   - The target variable was encoded using `LabelEncoder`, and categorical features were converted to dummy variables.

4. **Scaling**:
   - Features were standardized using `StandardScaler`.

5. **Model Training**:
   - Random Forest and Logistic Regression models were trained using `RandomizedSearchCV` for hyperparameter tuning.

6. **Evaluation**:
   - The best models were evaluated on the test set, and metrics such as accuracy, precision, recall, and F1-score were reported.

7. **Feature Importance**:
   - Feature importance was analyzed using the Random Forest model, and selected features were used to train a new model.

---

## References 
[1]: <https://archive.ics.uci.edu/dataset/320/student+performance> "Website"<br></br>


