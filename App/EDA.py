import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

def app():
    st.title("Exploratory Data Analysis (EDA)")

    # File mapping and format selection
    file_mapping = {"T20": "t20.csv", "ODI": "ODI data.csv", "Test": "test.csv"}
    format_choice = st.selectbox("Select Format:", list(file_mapping.keys()))
    data = pd.read_csv(file_mapping[format_choice])

    # Remove columns that include "Unnamed" in their names
    data = data.loc[:, ~data.columns.str.contains("Unnamed")]

    
    # Extract numeric data
    numeric_data = data.select_dtypes(include=["float64", "int64"])

    # Dataset Overview
    st.subheader(f"{format_choice} Dataset Overview")
    st.write(data.head())

    st.write("""
    **Interpretation:**
    - The dataset contains data on various cricket metrics for players across different formats. 
    - The columns represent different features such as matches played, runs scored, strike rate, etc.
    - The first few rows provide an overview of player stats.
    """)

    # Summary Statistics
    st.subheader("Summary Statistics")
    st.write(data.describe())

    st.write("""
    **Interpretation:**
    - **Count**: The total number of non-null entries for each column. This helps identify if there are missing values.
    - **Mean**: The average value for each feature. For example, the mean number of runs scored gives an idea of the average batting performance across the dataset.
    - **Std (Standard Deviation)**: Indicates how spread out the values are. A higher standard deviation suggests more variation in performance.
    - **Min and Max**: These give the range of values for each feature. They can show the lowest and highest performance for the given metrics.
    - **25%, 50%, 75%**: These are the quartiles that divide the data into four equal parts. The median is the 50% value and can be used to understand the central tendency.
    """)

    # Feature Distributions (Histograms)
    st.subheader("Feature Distributions")
    for column in data.select_dtypes(include=["float64", "int64"]):
        fig, ax = plt.subplots()
        sns.histplot(data[column], kde=True, ax=ax)
        ax.set_title(f"Distribution of {column}")
        st.pyplot(fig)

    st.write("""
    **Interpretation:**
    - Histograms display the frequency distribution of data, helping us understand how the values are spread.
    - If the data is skewed, it indicates an unequal distribution, which may require transformations (such as log transformation) for better modeling.
    - The **Kernel Density Estimation (KDE)** curve shows a smoothed version of the histogram, helping visualize the distribution more clearly.
    - For example, a cricket player's batting strike rate might be skewed to the right if most players have lower strike rates, but only a few players have very high strike rates.
    """)

    # Correlation Analysis
    st.subheader("Correlation Analysis")
    correlation = numeric_data.corr()
    st.write("Correlation Matrix:")
    st.dataframe(correlation.style.background_gradient(cmap="coolwarm", axis=None))

    st.write("""
    **Interpretation:**
    - **Correlation** shows how two variables are related. Values range from -1 to 1:
        - A correlation of 1 means a perfect positive relationship: when one variable increases, the other also increases.
        - A correlation of -1 means a perfect negative relationship: when one variable increases, the other decreases.
        - A correlation of 0 means no relationship.
    - Strong correlations (above 0.7 or below -0.7) suggest that one feature can help predict another.
    - For example, **runs scored** may have a strong positive correlation with **balls faced** (higher number of balls faced, higher the number of runs).
    - Correlations can guide us in selecting relevant features or identifying redundant features.
    """)

    # Correlation Heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
    st.pyplot(fig)

    st.write("""
    **Interpretation:**
    - The **heatmap** visually represents the correlation matrix with colors, making it easier to spot strong correlations.
    - The stronger the correlation, the more intense the color (either red for positive or blue for negative).
    - This heatmap helps to quickly identify relationships between features, such as whether batting average is strongly correlated with runs scored.
    """)

    # Skewness and Kurtosis Analysis
    st.subheader("Skewness and Kurtosis Analysis")
    skewness = data.select_dtypes(include=["float64", "int64"]).skew()
    kurt = data.select_dtypes(include=["float64", "int64"]).kurt()

    st.write("Skewness of Features:")
    st.write(skewness)

    st.write("Kurtosis of Features:")
    st.write(kurt)

    st.write("""
    **Interpretation:**
    - **Skewness** measures the asymmetry of the distribution. 
        - Positive skew indicates that the data tail is on the right side (higher values).
        - Negative skew indicates that the data tail is on the left side (lower values).
    - **Kurtosis** measures the "tailedness" of the data.
        - Positive kurtosis suggests that the data has heavy tails or more outliers.
        - Negative kurtosis suggests that the data has light tails.
    - Features with skewness near 0 and kurtosis near 3 are typically more normally distributed and easier to model.
    """)

    # Outlier Detection (Boxplots)
    st.subheader("Outlier Detection")
    for column in data.select_dtypes(include=["float64", "int64"]):
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=data[column], ax=ax, color="lightblue")
        ax.set_title(f"Boxplot for {column}")
        st.pyplot(fig)

    st.write("""
    **Interpretation:**
    - Boxplots visually show the distribution of data and help identify outliers (values that are far away from the rest of the data).
    - Outliers can indicate rare events or errors in data collection. For instance, an extremely high value for "runs scored" might indicate a player’s record-breaking performance.
    - Data with many outliers may need to be cleaned or transformed for modeling purposes.
    """)

    # Feature Importance (Correlation)
    st.subheader("Feature Importance")
    corr_threshold = 0.7  # You can modify this threshold based on your needs
    high_corr_features = correlation[correlation.abs() > corr_threshold].stack().index.tolist()
    
    st.write(f"Highly correlated features (|correlation| > {corr_threshold}):")
    st.write(high_corr_features)

    st.write("""
    **Interpretation:**
    - Highly correlated features indicate that two or more features are providing the same information.
    - You can consider removing one of these features or combining them into a single, more powerful feature.
    - This can help improve model performance by reducing multicollinearity.
    """)

    # Pairplot for Feature Relationships
    st.subheader("Pairplot of Features")
    sns.pairplot(data.select_dtypes(include=["float64", "int64"]))
    st.pyplot()

    st.write("""
    **Interpretation:**
    - Pairplots show scatterplots for every possible pair of features, helping to identify relationships, clusters, and trends between variables.
    - Diagonal plots display the distribution of each feature.
    - Pairplots also help in spotting potential outliers or unusual relationships between features.
    """)

    # Conclusion
    st.subheader("Conclusion")
    st.write("""
    - **Distributions**: Visualizing feature distributions helps us understand the data's central tendency, spread, and any skewness or outliers.
    - **Correlation Analysis**: Correlation analysis reveals relationships between features, which can guide us in feature selection.
    - **Skewness and Kurtosis**: Skewness and kurtosis provide insight into the shape of the distribution, which can impact modeling.
    - **Outlier Detection**: Boxplots help identify outliers, which can be significant or erroneous data points.
    - **Feature Importance**: Understanding highly correlated features helps in reducing redundancy and improving model performance.
    - **Pairplot**: The pairplot visually captures relationships between features and helps understand the overall data structure.
    """)

