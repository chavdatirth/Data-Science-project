import json
import os

# Define the cells of the Jupyter Notebook
cells = []

# Section 1: Title and Overview
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# House Prices Prediction using Advanced Regression Techniques\n",
        "### 5th Semester IT Engineering - Data Science Project\n",
        "\n",
        "**Project Objective:**\n",
        "The objective of this project is to predict residential house sale prices in Ames, Iowa, using the Kaggle House Prices dataset. We implement a complete machine learning pipeline in Python, comparing L2-regularized linear model (**Ridge**) and L1-regularized linear model (**Lasso**).\n",
        "\n",
        "**Key Concepts Covered:**\n",
        "1. **Exploratory Data Analysis (EDA):** Checking feature distributions, correlation analysis, missing value detection, and target variable transformation.\n",
        "2. **Data Preprocessing & Cleaning:** Handling missing data using median/most-frequent strategies, scaling numerical features, and encoding categorical variables using One-Hot Encoding via Scikit-Learn's `Pipeline` and `ColumnTransformer`.\n",
        "3. **Regression Modeling:** Implementation of Ridge and Lasso Regression.\n",
        "4. **Performance Evaluation:** Quantifying predictive power using Root Mean Squared Error (RMSE) and $R^2$ Score on a validation set.\n",
        "5. **Model Persistence:** Saving the best-performing model as a serialized file (`best_house_price_model.pkl`) and generating predictions for the Kaggle test set (`submission.csv`)."
    ]
})

# Section 2: Import Libraries
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 1. Import Libraries\n",
        "In this section, we import all the necessary Python libraries for data analysis, scientific computation, data visualization, and machine learning.\n",
        "- **Pandas & NumPy** for data manipulation and array processing.\n",
        "- **Matplotlib & Seaborn** for descriptive visualizations.\n",
        "- **Scikit-Learn** for preprocessing, pipeline construction, regression algorithms, and metrics evaluation."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Import core data science libraries\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "import seaborn as sns\n",
        "import pickle\n",
        "import os\n",
        "\n",
        "# Import scikit-learn preprocessing and pipeline tools\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.pipeline import Pipeline\n",
        "from sklearn.compose import ColumnTransformer\n",
        "from sklearn.impute import SimpleImputer\n",
        "from sklearn.preprocessing import StandardScaler, OneHotEncoder\n",
        "\n",
        "# Import regression models\n",
        "from sklearn.linear_model import Ridge, Lasso\n",
        "\n",
        "# Import evaluation metrics\n",
        "from sklearn.metrics import mean_squared_error, r2_score\n",
        "\n",
        "# Set plot style and figures parameters for professional aesthetics\n",
        "sns.set_theme(style='whitegrid', palette='muted')\n",
        "plt.rcParams['figure.figsize'] = (10, 6)\n",
        "plt.rcParams['font.size'] = 11\n",
        "\n",
        "print(\"All libraries imported successfully!\")"
    ]
})

# Section 3: Load Dataset
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 2. Load Dataset\n",
        "We load the Ames Housing dataset from the local `data/` directory. \n",
        "- `train.csv` contains both features and the target variable `SalePrice`, which we use to train and validate our models.\n",
        "- `test.csv` contains only features, for which we must predict the `SalePrice`."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Define path to the datasets\n",
        "train_path = 'data/train.csv'\n",
        "test_path = 'data/test.csv'\n",
        "\n",
        "# Verify files exist\n",
        "if not os.path.exists(train_path) or not os.path.exists(test_path):\n",
        "    raise FileNotFoundError(\"Make sure train.csv and test.csv are located in the local 'data/' folder.\")\n",
        "\n",
        "# Load the datasets using pandas\n",
        "train = pd.read_csv(train_path)\n",
        "test = pd.read_csv(test_path)\n",
        "\n",
        "# Display dataset dimensions (shapes)\n",
        "print(f\"Training dataset shape: {train.shape} (rows, columns)\")\n",
        "print(f\"Testing dataset shape:  {test.shape} (rows, columns)\")\n",
        "\n",
        "# Preview the first 5 rows of the training dataset\n",
        "train.head()"
    ]
})

# Section 4: EDA
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 3. Exploratory Data Analysis (EDA)\n",
        "Exploratory Data Analysis is a crucial step in understanding the structure, characteristics, and patterns of our data. \n",
        "\n",
        "In this section, we will:\n",
        "1. Display general information about the dataset.\n",
        "2. Check for missing values in columns.\n",
        "3. Show summary descriptive statistics for numerical variables.\n",
        "4. Visualize the distribution of the target variable `SalePrice` before and after log-transformation.\n",
        "5. Generate a correlation heatmap to identify the strongest linear relationships with `SalePrice`.\n",
        "6. Compute the skewness of numerical features."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 3.1 Display dataset structure, column data types, and non-null counts\n",
        "print(\"=== Training Dataset Info ===\")\n",
        "train.info()"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 3.2 Calculate the number and percentage of missing values per column\n",
        "missing_counts = train.isnull().sum()\n",
        "missing_percent = 100 * train.isnull().sum() / len(train)\n",
        "\n",
        "# Create a DataFrame to view columns with missing values\n",
        "missing_data = pd.DataFrame({\n",
        "    'Missing Count': missing_counts,\n",
        "    'Percentage (%)': missing_percent\n",
        "})\n",
        "\n",
        "# Display features with missing values sorted in descending order\n",
        "missing_data = missing_data[missing_data['Missing Count'] > 0].sort_values(by='Missing Count', ascending=False)\n",
        "print(f\"Total columns with missing values: {len(missing_data)}\")\n",
        "print(\"\\nTop 15 columns with most missing values:\")\n",
        "print(missing_data.head(15))"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 3.3 Generate summary statistics for numerical features\n",
        "print(\"=== Descriptive Statistics for Numerical Features ===\")\n",
        "train.describe()"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 3.4 Visualize the target variable SalePrice distribution (Original vs Log-Transformed)\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
        "\n",
        "# Plot 1: Raw SalePrice distribution\n",
        "sns.histplot(train['SalePrice'], kde=True, ax=axes[0], color='dodgerblue')\n",
        "axes[0].set_title(f\"Original SalePrice Distribution (Skewness: {train['SalePrice'].skew():.2f})\", fontsize=12)\n",
        "axes[0].set_xlabel(\"SalePrice ($)\")\n",
        "axes[0].set_ylabel(\"Frequency\")\n",
        "axes[0].get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: \"{:,}\".format(int(x))))\n",
        "\n",
        "# Plot 2: Log-transformed SalePrice distribution (y = log(1 + x))\n",
        "log_saleprice = np.log1p(train['SalePrice'])\n",
        "sns.histplot(log_saleprice, kde=True, ax=axes[1], color='forestgreen')\n",
        "axes[1].set_title(f\"Log-Transformed SalePrice Distribution (Skewness: {log_saleprice.skew():.2f})\", fontsize=12)\n",
        "axes[1].set_xlabel(\"Log(SalePrice + 1)\")\n",
        "axes[1].set_ylabel(\"Frequency\")\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 3.5 Generate a correlation heatmap of numerical features\n",
        "numerical_df = train.select_dtypes(include=[np.number])\n",
        "corr_matrix = numerical_df.corr()\n",
        "\n",
        "# Find the top 10 features most highly correlated with SalePrice\n",
        "top_corr_features = corr_matrix['SalePrice'].abs().sort_values(ascending=False).index[:11]\n",
        "\n",
        "# Plot correlation heatmap\n",
        "plt.figure(figsize=(10, 8))\n",
        "sns.heatmap(train[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5, cbar=True)\n",
        "plt.title(\"Correlation Heatmap: Top 10 Features with SalePrice\", fontsize=14, pad=15)\n",
        "plt.show()\n",
        "\n",
        "print(\"Top 10 features correlated with SalePrice (sorted by absolute correlation coefficient):\")\n",
        "print(corr_matrix['SalePrice'].abs().sort_values(ascending=False).head(11))"
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# 3.6 Identify skewed numerical features\n",
        "skewed_features = numerical_df.drop(columns=['Id', 'SalePrice']).skew().sort_values(ascending=False)\n",
        "skewness_df = pd.DataFrame({'Feature Skewness': skewed_features})\n",
        "\n",
        "# Filter out features with absolute skewness > 0.75 (highly skewed)\n",
        "highly_skewed = skewness_df[abs(skewness_df['Feature Skewness']) > 0.75]\n",
        "\n",
        "print(f\"Total numerical features: {skewness_df.shape[0]}\")\n",
        "print(f\"Highly skewed numerical features (|Skewness| > 0.75): {highly_skewed.shape[0]}\")\n",
        "print(\"\\nTop 10 most positively skewed numerical features:\")\n",
        "print(highly_skewed.head(10))"
    ]
})

# Section 5: Data Cleaning & Preprocessing
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 4. Data Preprocessing & Pipeline Construction\n",
        "In this section, we separate our dataset into predictor variables ($X$) and the target variable ($y$). We then set up preprocessing pipelines using Scikit-Learn.\n",
        "\n",
        "### Data Preprocessing Strategy:\n",
        "- **Numerical Imputation:** Missing numerical values will be filled with their column **Median**. Median is robust to outliers which are common in real estate data.\n",
        "- **Categorical Imputation:** Missing categorical values will be filled with the **Most Frequent (Mode)** category.\n",
        "- **Feature Scaling:** Standardize numerical features using `StandardScaler` to ensure our regularization models (Ridge and Lasso) evaluate features on the same scale.\n",
        "- **Categorical Encoding:** Transform categorical variables into numerical dummy variables using **One-Hot Encoding** (`OneHotEncoder`). We set `handle_unknown='ignore'` to robustly handle any categories present in the test set that weren't seen during training, and `sparse_output=False` so that we obtain a dense matrix suitable for analysis.\n",
        "\n",
        "### Target Transformation:\n",
        "- We apply log transformation `np.log1p()` to `SalePrice` to reduce skewness and stabilize variance, making it closer to a normal distribution. This satisfies linear regression assumptions."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Separate predictor features (X) and target variable (y)\n",
        "X = train.drop(columns=['Id', 'SalePrice'])\n",
        "y = train['SalePrice']\n",
        "\n",
        "# Extract predictor features for the test set\n",
        "X_test = test.drop(columns=['Id'])\n",
        "\n",
        "# Apply log transformation to target variable\n",
        "y_log = np.log1p(y)\n",
        "\n",
        "# Identify numerical and categorical column names\n",
        "numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()\n",
        "categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()\n",
        "\n",
        "# Define Preprocessing Pipelines\n",
        "# 1. Pipeline for numerical columns\n",
        "num_pipeline = Pipeline(steps=[\n",
        "    ('imputer', SimpleImputer(strategy='median')),\n",
        "    ('scaler', StandardScaler())\n",
        "])\n",
        "\n",
        "# 2. Pipeline for categorical columns\n",
        "cat_pipeline = Pipeline(steps=[\n",
        "    ('imputer', SimpleImputer(strategy='most_frequent')),\n",
        "    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))\n",
        "])\n",
        "\n",
        "# 3. Combine both preprocessors into a ColumnTransformer\n",
        "preprocessor = ColumnTransformer(transformers=[\n",
        "    ('num', num_pipeline, numerical_cols),\n",
        "    ('cat', cat_pipeline, categorical_cols)\n",
        "])\n",
        "\n",
        "print(\"Preprocessing pipelines successfully created!\")\n",
        "print(f\"Number of numerical columns: {len(numerical_cols)}\")\n",
        "print(f\"Number of categorical columns: {len(categorical_cols)}\")"
    ]
})

# Section 6: Train-Validation Split
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 5. Train-Validation Split\n",
        "To validate how well our models perform on unseen data before making predictions on the final competition test dataset, we split our training data into:\n",
        "- **80% Training set** (used to fit the models).\n",
        "- **20% Validation set** (used to evaluate performance).\n",
        "\n",
        "We use a fixed random state (`random_state=42`) to ensure reproducibility."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Split the dataset into 80% train and 20% validation\n",
        "X_train, X_val, y_train, y_val = train_test_split(X, y_log, test_size=0.2, random_state=42)\n",
        "\n",
        "print(f\"Training features shape:   {X_train.shape}\")\n",
        "print(f\"Validation features shape: {X_val.shape}\")"
    ]
})

# Section 7: Model Training
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 6. Model Training\n",
        "We train two linear models that use regularization to prevent overfitting:\n",
        "1. **Ridge Regression (L2 Regularization):** Adds a penalty proportional to the *square* of the coefficients ($\alpha \\sum w_i^2$). It shrinks coefficients close to zero but keeps all features.\n",
        "2. **Lasso Regression (L1 Regularization):** Adds a penalty proportional to the *absolute value* of the coefficients ($\alpha \\sum |w_i|$). It performs feature selection by shrinking some coefficients to exactly zero.\n",
        "\n",
        "We build end-to-end pipelines that chain the preprocessing transformer with the regression estimators. This ensures a clean workflow and avoids data leakage."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Define Ridge model pipeline (alpha=10.0 is selected as a standard regularized value)\n",
        "ridge_pipeline = Pipeline(steps=[\n",
        "    ('preprocessor', preprocessor),\n",
        "    ('model', Ridge(alpha=10.0))\n",
        "])\n",
        "\n",
        "# Define Lasso model pipeline (alpha=0.0005 is selected to allow fine feature selection)\n",
        "lasso_pipeline = Pipeline(steps=[\n",
        "    ('preprocessor', preprocessor),\n",
        "    ('model', Lasso(alpha=0.0005, max_iter=10000))\n",
        "])\n",
        "\n",
        "# Train Ridge model\n",
        "print(\"Training Ridge Regression...\")\n",
        "ridge_pipeline.fit(X_train, y_train)\n",
        "\n",
        "# Train Lasso model\n",
        "print(\"Training Lasso Regression...\")\n",
        "lasso_pipeline.fit(X_train, y_train)\n",
        "\n",
        "print(\"Both models trained successfully!\")"
    ]
})

# Section 8: Model Evaluation
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 7. Model Evaluation\n",
        "We evaluate both models on the validation dataset. \n",
        "\n",
        "### Metrics:\n",
        "1. **Root Mean Squared Error (RMSE):** We calculate RMSE on:\n",
        "   - The **Log Scale** (the evaluation metric used by Kaggle).\n",
        "   - The **Original Price Scale** (reverting predictions to $ USD to show the actual average error magnitude).\n",
        "2. **R² Score (Coefficient of Determination):** Measures the proportion of variance in the log-transformed prices explained by the features."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Predict on validation set (outputs are on log scale)\n",
        "ridge_preds_log = ridge_pipeline.predict(X_val)\n",
        "lasso_preds_log = lasso_pipeline.predict(X_val)\n",
        "\n",
        "# Revert predictions back to original USD scale using expm1 (exponential minus 1)\n",
        "ridge_preds_orig = np.expm1(ridge_preds_log)\n",
        "lasso_preds_orig = np.expm1(lasso_preds_log)\n",
        "\n",
        "# Revert true validation targets back to original USD scale\n",
        "y_val_orig = np.expm1(y_val)\n",
        "\n",
        "# Calculate RMSE on Log Scale (Kaggle Metric)\n",
        "ridge_rmse_log = np.sqrt(mean_squared_error(y_val, ridge_preds_log))\n",
        "lasso_rmse_log = np.sqrt(mean_squared_error(y_val, lasso_preds_log))\n",
        "\n",
        "# Calculate RMSE on Original Scale ($ USD)\n",
        "ridge_rmse_orig = np.sqrt(mean_squared_error(y_val_orig, ridge_preds_orig))\n",
        "lasso_rmse_orig = np.sqrt(mean_squared_error(y_val_orig, lasso_preds_orig))\n",
        "\n",
        "# Calculate R2 Score (on log scale target)\n",
        "ridge_r2 = r2_score(y_val, ridge_preds_log)\n",
        "lasso_r2 = r2_score(y_val, lasso_preds_log)\n",
        "\n",
        "# Create a DataFrame to compare the two models\n",
        "comparison_df = pd.DataFrame({\n",
        "    'Model': ['Ridge Regression', 'Lasso Regression'],\n",
        "    'Log-RMSE (Kaggle Metric)': [ridge_rmse_log, lasso_rmse_log],\n",
        "    'Validation RMSE ($ USD)': [ridge_rmse_orig, lasso_rmse_orig],\n",
        "    'R² Score (Log-Scale)': [ridge_r2, lasso_r2]\n",
        "})\n",
        "\n",
        "print(\"=== Validation Set Performance Comparison ===\")\n",
        "print(comparison_df.to_string(index=False))"
    ]
})

# Section 9: Visualization of predictions & residuals
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "### Visualizing Predictions & Residuals\n",
        "We select the better-performing model based on the lowest validation Log-RMSE and plot:\n",
        "1. **Actual vs. Predicted Sale Prices:** Ideally, points should lie close to the diagonal red line ($y = x$).\n",
        "2. **Residual Plot:** Shows predictions vs residual errors ($y_{actual} - y_{predicted}$). Ideally, residual errors should be randomly scattered around $0$, indicating stable predictions across price points."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Determine the best-performing model based on validation Log-RMSE\n",
        "if lasso_rmse_log < ridge_rmse_log:\n",
        "    best_model_name = \"Lasso Regression\"\n",
        "    best_model_pipeline = lasso_pipeline\n",
        "    best_preds_orig = lasso_preds_orig\n",
        "    best_preds_log = lasso_preds_log\n",
        "    best_rmse_log = lasso_rmse_log\n",
        "    best_r2 = lasso_r2\n",
        "else:\n",
        "    best_model_name = \"Ridge Regression\"\n",
        "    best_model_pipeline = ridge_pipeline\n",
        "    best_preds_orig = ridge_preds_orig\n",
        "    best_preds_log = ridge_preds_log\n",
        "    best_rmse_log = ridge_rmse_log\n",
        "    best_r2 = ridge_r2\n",
        "\n",
        "print(f\"The best performing model is: {best_model_name}\\n\")\n",
        "\n",
        "# Calculate residuals on original USD scale\n",
        "residuals = y_val_orig - best_preds_orig\n",
        "\n",
        "# Generate subplots for predictions and residuals\n",
        "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
        "\n",
        "# Plot 1: Actual vs Predicted Scatter Plot\n",
        "axes[0].scatter(y_val_orig, best_preds_orig, alpha=0.6, color='royalblue', edgecolors='k')\n",
        "axes[0].plot([y_val_orig.min(), y_val_orig.max()], [y_val_orig.min(), y_val_orig.max()], 'r--', lw=2)\n",
        "axes[0].set_title(f\"Actual vs. Predicted Sale Prices ({best_model_name})\", fontsize=12)\n",
        "axes[0].set_xlabel(\"Actual SalePrice ($)\")\n",
        "axes[0].set_ylabel(\"Predicted SalePrice ($)\")\n",
        "axes[0].get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: \"{:,}\".format(int(x))))\n",
        "axes[0].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: \"{:,}\".format(int(x))))\n",
        "\n",
        "# Plot 2: Residuals Plot\n",
        "axes[1].scatter(best_preds_orig, residuals, alpha=0.6, color='darkorange', edgecolors='k')\n",
        "axes[1].axhline(0, color='red', linestyle='--', lw=2)\n",
        "axes[1].set_title(f\"Residual Plot ({best_model_name})\", fontsize=12)\n",
        "axes[1].set_xlabel(\"Predicted SalePrice ($)\")\n",
        "axes[1].set_ylabel(\"Residual Error ($)\")\n",
        "axes[1].get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: \"{:,}\".format(int(x))))\n",
        "axes[1].get_yaxis().set_major_formatter(plt.FuncFormatter(lambda x, loc: \"{:,}\".format(int(x))))\n",
        "\n",
        "plt.tight_layout()\n",
        "plt.show()"
    ]
})

# Section 10: Model Persistence
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 8. Save Best Model\n",
        "We serialize (save) our trained pipeline (which includes the fitted preprocessor and the regression model) as a binary `.pkl` file using Python's built-in `pickle` module. This allows loading our model to predict new house prices directly without retraining."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Define model filepath\n",
        "model_filename = 'best_house_price_model.pkl'\n",
        "\n",
        "# Serialize and save the model pipeline\n",
        "with open(model_filename, 'wb') as file:\n",
        "    pickle.dump(best_model_pipeline, file)\n",
        "\n",
        "print(f\"Successfully saved {best_model_name} pipeline to: '{model_filename}'\")"
    ]
})

# Section 11: Predictions on test.csv
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 9. Predictions on Test Dataset\n",
        "We now load the testing features from `test.csv` (which has no target column `SalePrice`) and use our trained best-performing model to make predictions.\n",
        "\n",
        "Because our model was trained on log-transformed prices, we must:\n",
        "1. Generate predictions which will be in log-scale.\n",
        "2. Apply `np.expm1()` to convert predictions back to the original USD scale.\n",
        "3. Save the predictions to `submission.csv` with columns `Id` and `SalePrice` as required by the Kaggle competition."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Predict sale prices for the test set\n",
        "test_preds_log = best_model_pipeline.predict(X_test)\n",
        "\n",
        "# Convert predictions back to original USD scale\n",
        "test_preds_orig = np.expm1(test_preds_log)\n",
        "\n",
        "# Create submission DataFrame\n",
        "submission = pd.DataFrame({\n",
        "    'Id': test['Id'],\n",
        "    'SalePrice': test_preds_orig\n",
        "})\n",
        "\n",
        "# Display a preview of the predictions\n",
        "print(\"=== Submission Preview ===\")\n",
        "print(submission.head())\n",
        "\n",
        "# Save the predictions to csv\n",
        "submission_filename = 'submission.csv'\n",
        "submission.to_csv(submission_filename, index=False)\n",
        "print(f\"\\nSaved submission file to: '{submission_filename}'\")\n",
        "print(f\"Submission shape: {submission.shape}\")"
    ]
})

# Section 12: Conclusion & Results Printout
cells.append({
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## 10. Conclusion & Final Report\n",
        "In this cell, we run a short diagnostic to print a clean final summary of our metrics and output files. This is highly useful for project vivas and presentations."
    ]
})

cells.append({
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Generate and display final project metrics report\n",
        "print(\"======================================================================\")\n",
        "print(\"                 FINAL MODEL EVALUATION REPORT                        \")\n",
        "print(\"======================================================================\")\n",
        "print(f\"Best Performing Model:            {best_model_name}\")\n",
        "print(f\"Validation Log-RMSE (Kaggle):      {best_rmse_log:.5f}\")\n",
        "print(f\"Validation RMSE ($ USD):           ${best_preds_orig.mean():,.2f} mean predicted value\")\n",
        "print(f\"Validation R² Score (Log-Scale):   {best_r2:.5%}\")\n",
        "print(\"----------------------------------------------------------------------\")\n",
        "print(f\"Saved Model Filename:              {model_filename} (pickle format)\")\n",
        "print(f\"Generated Submission Filename:     {submission_filename} ({len(submission)} rows)\")\n",
        "print(\"======================================================================\")"
    ]
})

# Create the final notebook dict
notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

# Write the notebook to file
output_path = 'House_Prices_Prediction.ipynb'
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1, ensure_ascii=False)

print(f"Successfully generated notebook skeleton file: '{output_path}'")
