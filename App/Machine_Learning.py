import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
import seaborn as sns

def app():
    st.title("Enhanced Machine Learning Model with Interpretations")
    file_mapping = {"T20": "t20.csv", "ODI": "ODI data.csv", "Test": "test.csv"}

    # File selection
    format_choice = st.selectbox("Select Format:", list(file_mapping.keys()))
    data = pd.read_csv(file_mapping[format_choice], na_values=['-', 'NA', ''])

    # Remove columns that include "Unnamed" in their names
    data = data.loc[:, ~data.columns.str.contains("Unnamed")]

    # Dataset Overview
    st.subheader(f"{format_choice} Dataset Overview")
    st.write(f"Dataset Dimensions: {data.shape[0]} rows and {data.shape[1]} columns")
    st.write("Data Types:")
    st.write(data.dtypes)
    st.write("Missing Values per Column:")
    st.write(data.isnull().sum())

    # Show dataset preview
    st.write(data.head())
    st.write("Summary Statistics:")
    st.write(data.describe())

    st.write("""
    **Interpretation:**
    - The dataset has several columns representing various batting metrics like matches played, innings, not-outs, balls faced, and strike rate.
    - The missing values and data types are shown for your review. We will handle missing data in our model to ensure accurate predictions.
    """)

    # Selecting Features
    features = ["Mat", "Inns", "NO", "BF", "SR"]
    target = "Runs"
    features = [feature for feature in features if feature in data.columns]

    # Normalize column names for consistency
    data.columns = data.columns.str.strip().str.lower()  # Remove whitespace and convert to lowercase
    features = [feature.lower() for feature in features]  # Normalize features list
    target = target.lower()  # Normalize target name

    if all(feature in data.columns for feature in features + [target]):
        # Prepare data
        X = data[features]
        y = data[target]

        # Handle missing values
        X.fillna(0, inplace=True)  # Replace NaN in features with 0
        y.fillna(y.mean(), inplace=True)  # Replace NaN in target with mean

        # Categorical Encoding (if applicable)
        categorical_columns = X.select_dtypes(include=['object']).columns
        if not categorical_columns.empty:
            st.write("Categorical features detected. Applying one-hot encoding.")
            X = pd.get_dummies(X, columns=categorical_columns, drop_first=True)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Check for multicollinearity
        st.subheader("Feature Correlations")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(X.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        st.write("""
        **Interpretation:**
        - The correlation heatmap shows how strongly the features are related to each other.
        - Strong correlations (positive or negative) suggest that some features might have similar information, which can affect model performance. In these cases, it's essential to carefully select features to avoid redundancy.
        """)

        # Train model
        model = RandomForestRegressor(random_state=42)
        model.fit(X_train, y_train)

        # Predictions
        predictions = model.predict(X_test)

        # Metrics
        mse = mean_squared_error(y_test, predictions)
        mae = mean_absolute_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        adj_r2 = 1 - (1 - r2) * (len(y_test) - 1) / (len(y_test) - X_test.shape[1] - 1)

        st.subheader("Model Performance")
        st.write(f"Mean Squared Error (MSE): {mse:.2f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"R² Score: {r2:.2f}")
        st.write(f"Adjusted R² Score: {adj_r2:.2f}")

        st.write("""
        **Interpretation:**
        - **Mean Squared Error (MSE)**: Measures how far off the predictions are from the actual values. Lower values are better.
        - **Mean Absolute Error (MAE)**: Shows the average magnitude of errors between predicted and actual values. Lower values indicate better performance.
        - **R² Score**: Indicates the proportion of variance in the target variable explained by the model. A higher R² means a better fit.
        - **Adjusted R² Score**: Corrects the R² score for the number of features. This is helpful to prevent overfitting when adding more features.
        """)

        # Feature Importance
        st.subheader("Feature Importance")
        feature_importance = pd.DataFrame({
            "Feature": X.columns,
            "Importance": model.feature_importances_
        }).sort_values(by="Importance", ascending=False)
        st.write(feature_importance)

        st.write("""
        **Interpretation:**
        - Feature importance tells us which factors are most influential in predicting a player's runs.
        - The features with the highest importance values are the most predictive of the target variable, 'Runs'.
        """)

        # Plot feature importance
        fig, ax = plt.subplots()
        ax.barh(feature_importance["Feature"], feature_importance["Importance"], color="skyblue")
        ax.set_xlabel("Importance")
        ax.set_title("Feature Importance")
        st.pyplot(fig)

        # EDA: Pairplot
        st.subheader("Exploratory Data Analysis")
        st.write("Pairplot of Selected Features and Target")
        pairplot_data = data[features + [target]]
        sns.pairplot(pairplot_data)
        st.pyplot()

        st.write("""
        **Interpretation:**
        - The pairplot helps visualize the relationships between the selected features and the target variable.
        - Strong linear relationships between features and the target can suggest useful predictors for the model.
        """)

        # Plot predictions vs. actual values
        st.subheader("Predictions vs Actual Values")
        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions, alpha=0.6)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--")
        ax.set_xlabel("Actual Values")
        ax.set_ylabel("Predicted Values")
        ax.set_title("Predictions vs Actual Values")
        st.pyplot(fig)

        st.write("""
        **Interpretation:**
        - The scatter plot shows the actual values against the predicted values. 
        - Ideally, the points should lie on the red dashed line, indicating that the model's predictions are close to the actual values.
        - The spread of points away from the line indicates errors in prediction.
        """)

        # Residual Plot
        st.subheader("Residual Analysis")
        residuals = y_test - predictions
        fig, ax = plt.subplots()
        ax.scatter(predictions, residuals, alpha=0.6)
        ax.axhline(0, color="red", linestyle="--")
        ax.set_xlabel("Predicted Values")
        ax.set_ylabel("Residuals")
        ax.set_title("Residuals vs Predicted Values")
        st.pyplot(fig)

        st.write("""
        **Interpretation:**
        - The residual plot shows the difference between predicted and actual values (residuals).
        - Ideally, residuals should be randomly scattered around zero with no distinct patterns. A pattern could indicate model bias.
        """)

        # Histogram of Target Variable
        st.subheader("Target Variable Distribution")
        fig, ax = plt.subplots()
        ax.hist(y, bins=20, color="lightblue", edgecolor="black")
        ax.set_xlabel("Target (Runs)")
        ax.set_ylabel("Frequency")
        ax.set_title("Distribution of Target Variable")
        st.pyplot(fig)

        st.write("""
        **Interpretation:**
        - The histogram shows the distribution of the target variable ('Runs').
        - Understanding the distribution helps assess whether the model may benefit from additional feature engineering, such as handling skewed data.
        """)

    else:
        st.error("The dataset is missing one or more required features or the target variable.")
        st.write("""
        **Interpretation:**
        - This message appears when the selected dataset does not contain all the required features for the model.
        - Make sure that the dataset includes all necessary columns like 'Mat', 'Inns', 'NO', 'BF', 'SR', and the target 'Runs'.
        """)
