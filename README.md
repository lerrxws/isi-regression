
# **Overview**

This script implements a Ridge Regression model for supervised learning tasks. It includes preprocessing of data, hyperparameter tuning using cross-validation, evaluation of the model, and visualization of the results. The output consists of the predicted values for evaluation data and other key files for analysis.

---

## **Key Features**

1. **Data Preprocessing:**
   - Handles numerical and categorical data using pipelines.
   - Applies standardization to numerical features.
   - Uses one-hot encoding for categorical features.

2. **Model Training:**
   - Utilizes Ridge Regression for prediction.
   - Optimizes model parameters using GridSearchCV.

3. **Evaluation:**
   - Measures model performance using R-squared (R²) and Mean Squared Error (MSE) metrics.

4. **Visualization:**
   - Visualizes the impact of hyperparameters on model performance.
   - Generates scatter plots of the given `y` values and saves visualizations.

5. **Outputs:**
   - Saves the best parameters and scores as a CSV.
   - Exports visualizations of hyperparameter effects.
   - Outputs the predictions for evaluation data.

---

## **File Inputs**

1. `input/X_public.npy`  
   - Feature set for training and validation.

2. `input/y_public.npy`  
   - Target labels for training and validation.

3. `input/X_eval.npy`  
   - Features for evaluation (test) set.

---

## **Generated Outputs**

1. **Predictions:**
   - `output/y_predikcia.npy`: Predicted values for the evaluation dataset.

2. **Visualization Files:**
   - Plots showing the impact of hyperparameters on the model's performance are saved in `output/images/`.

3. **Best Cross-Validation Results:**
   - `output/best_cv_results.csv`: Contains the best parameters and corresponding score.

---

## **Code Structure**

1. **Data Preprocessing:**
   - Categorical and numerical features are processed using a `ColumnTransformer`.

2. **Model Training:**
   - A Ridge Regression model is trained with multiple hyperparameter combinations using `GridSearchCV`.

3. **Visualization:**
   - The script generates plots to analyze the effect of individual hyperparameters on model performance.

4. **Saving Results:**
   - Saves predictions and best model parameters for further analysis.


