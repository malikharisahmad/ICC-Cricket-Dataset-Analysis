import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis

# Set page configuration at the very start
st.set_page_config(page_title="Cricket Data Analysis", layout="wide")

def app():
    # Title of the page
    st.title("Statistical Analysis of Cricket Data")

    # File mapping and selection
    file_mapping = {"T20": "t20.csv", "ODI": "ODI data.csv", "Test": "test.csv"}
    format_choice = st.selectbox("Select Format:", list(file_mapping.keys()), index=0)

    # Load data
    data = pd.read_csv(file_mapping[format_choice])

    # Remove columns that include "Unnamed" in their names
    data = data.loc[:, ~data.columns.str.contains("Unnamed")]

    # Extract numeric data
    numeric_data = data.select_dtypes(include=["float64", "int64"])

    # Descriptive statistics
    st.subheader(f"Descriptive Statistics for {format_choice}")
    st.write(numeric_data.describe().style.set_table_styles(
        [{'selector': 'thead th', 'props': [('background-color', '#2e3d49'), ('color', 'white')]}, 
         {'selector': 'tbody td', 'props': [('background-color', '#f5f5f5'), ('color', 'black')]}, 
         {'selector': 'tr:nth-child(even)', 'props': [('background-color', '#f9f9f9')]}, 
         {'selector': 'tr:nth-child(odd)', 'props': [('background-color', '#ffffff')]}]
    ))

    st.write("""
    **Interpretation:**
    - **Count**: Shows the number of non-null entries for each feature. It's important for checking missing data.
    - **Mean**: The average value of the feature. For example, the mean number of matches played (Mat) can give a general idea of player involvement.
    - **Standard Deviation (std)**: Measures the spread or variability of the data. Higher values indicate more variation among player performances.
    - **Min & Max**: These represent the range of values for each feature. For example, the minimum and maximum number of runs scored can indicate the range of batting performances.
    - **25%, 50%, 75%**: These are quartiles that split the data into four parts. The 50% is the median, which is the middle value when data is sorted.
    """)

    # Correlation Analysis
    st.subheader("Correlation Analysis")
    correlation = numeric_data.corr()
    st.write("Correlation Matrix:")
    st.dataframe(correlation.style.background_gradient(cmap="coolwarm", axis=None))

    st.write("""
    **Interpretation:**
    - Correlation measures the relationship between two variables. A positive value means the features move in the same direction, while a negative value indicates they move in opposite directions.
    - For example, a high positive correlation between "Inns" (Innings) and "Runs" suggests that players who play more innings tend to score more runs.
    - A value close to 0 means little to no correlation between the two features.
    - Strong correlations are important when building models because we might use them to select features.
    """)

    # Highlight significant correlations
    high_corr = correlation[(correlation > 0.7) & (correlation < 1)]
    low_corr = correlation[correlation < -0.7]

    with st.expander("Highly Positive Correlations (> 0.7)"):
        st.write(high_corr.dropna(how="all").dropna(axis=1, how="all"))
    with st.expander("Highly Negative Correlations (< -0.7)"):
        st.write(low_corr.dropna(how="all").dropna(axis=1, how="all"))

    # Correlation heatmap
    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", fmt=".2f", ax=ax, linewidths=0.5)
    st.pyplot(fig)

    st.write("""
    **Interpretation of the Heatmap:**
    - The color scale represents the strength of the correlation. Red shows strong positive correlations, and blue shows strong negative correlations.
    - You can use the heatmap to visually identify which features are strongly related and may therefore be useful together for predictions.
    """)

    # Outlier detection
    st.subheader("Outlier Detection")
    outlier_dict = {}  # Dictionary to store outliers for display

    for column in numeric_data.columns:
        q1 = numeric_data[column].quantile(0.25)
        q3 = numeric_data[column].quantile(0.75)
        iqr = q3 - q1
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        outliers = numeric_data[(numeric_data[column] < lower_bound) | (numeric_data[column] > upper_bound)]

        outlier_dict[column] = outliers[[column]]  # Store outliers for the column

    for column, outliers in outlier_dict.items():
        with st.expander(f"Outliers in {column}"):
            st.write(outliers.style.set_properties(**{'background-color': 'yellow', 'color': 'black'}))

    st.write("""
    **Interpretation:**
    - Outliers are extreme values that deviate significantly from the rest of the data. They may result from data errors or represent rare but valid occurrences.
    - Identifying and addressing outliers can help improve the accuracy of machine learning models, as outliers can disproportionately affect model performance.
    """)

    # Outlier Visualization
    st.subheader("Outlier Visualization (Boxplots)")
    for column in numeric_data.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.boxplot(x=numeric_data[column], ax=ax, color="lightblue")
        ax.set_title(f"Boxplot for {column}")
        st.pyplot(fig)

    st.write("""
    **Interpretation of Boxplots:**
    - Boxplots visually show the distribution of data, highlighting the median, quartiles, and potential outliers.
    - The "whiskers" show the range of data within 1.5 times the interquartile range, and any points outside this range are considered outliers.
    """)

    # Covariance Matrix
    st.subheader("Covariance Matrix")
    covariance = numeric_data.cov()
    st.write(covariance.style.background_gradient(cmap="Blues"))

    st.write("""
    **Interpretation:**
    - Covariance measures the relationship between two variables, similar to correlation, but without normalization.
    - Positive covariance indicates that the two variables move in the same direction, while negative covariance indicates they move in opposite directions.
    - The magnitude of covariance is not standardized, so the scale depends on the units of the variables.
    """)

    # Feature Variability
    st.subheader("Feature Variability (Standard Deviation)")
    variability = numeric_data.std()
    variability_df = variability.to_frame(name="Standard Deviation")  # Convert Series to DataFrame
    st.write(variability_df.style.highlight_max(axis=0, color="lightgreen"))

    st.write("""
    **Interpretation:**
    - The standard deviation shows how spread out the values are for each feature.
    - Features with higher variability (larger standard deviation) can provide more information, as they indicate a wider range of values.
    - Features with low variability may not contribute much useful information to models and could be dropped in feature selection.
    """)

    # Feature Distribution - Histograms
    st.subheader("Feature Distributions")
    for column in numeric_data.columns:
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.histplot(numeric_data[column], kde=True, ax=ax, color="skyblue", bins=20)
        ax.set_title(f"Distribution of {column}")
        st.pyplot(fig)

    st.write("""
    **Interpretation of Histograms:**
    - Histograms show the frequency distribution of a feature's values.
    - The shape of the histogram can indicate the underlying distribution, such as normal, skewed, or bimodal distributions.
    - Understanding the distribution can help in deciding whether a transformation is needed (e.g., log transformation for highly skewed data).
    """)

    # Skewness and Kurtosis
    st.subheader("Skewness and Kurtosis Analysis")
    skewness = numeric_data.skew()
    kurt = numeric_data.kurt()
    
    skewness_df = pd.DataFrame(skewness, columns=["Skewness"])
    kurt_df = pd.DataFrame(kurt, columns=["Kurtosis"])
    
    st.write("Skewness of Features:")
    st.write(skewness_df.style.highlight_max(axis=0, color="lightgreen"))
    
    st.write("Kurtosis of Features:")
    st.write(kurt_df.style.highlight_max(axis=0, color="lightgreen"))
    
    st.write("""
    **Interpretation:**
    - **Skewness**: If skewness is close to 0, the data is symmetric. Positive skewness means the data is right-skewed, and negative skewness means left-skewed.
    - **Kurtosis**: High kurtosis means the data has heavy tails or outliers, and low kurtosis indicates lighter tails. A normal distribution has a kurtosis of 3.
    """)

    st.write("""
    **Conclusion:**
    - The correlation matrix and covariance matrix help us understand the relationships between features, which is essential for feature selection.
    - Outliers can distort model performance, so detecting and addressing them is crucial.
    - Skewness and kurtosis analysis can help identify whether data transformation (like normalization) is needed before modeling.
    """)

