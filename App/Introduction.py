import streamlit as st

def app():
    st.title("Introduction to the Cricket Dataset")

    st.write("""
    Welcome to the **Cricket Dataset Explorer**! This application provides insights into cricket batting statistics across different formats, including **T20**, **ODI**, and **Test**. The dataset captures various metrics related to player performance, helping you analyze trends in batting performances and compare players' statistics in different formats.

    Whether you're a cricket fan, analyst, or data enthusiast, this dataset offers valuable insights into player performance and can serve as a foundation for deeper statistical analyses and predictive modeling.
    """)

    st.markdown("### Dataset Overview")
    st.write("""
    The dataset contains detailed batting statistics for players across **three formats** of cricket: **T20**, **ODI**, and **Test**. These formats vary in terms of match length and playing conditions, which often lead to different player performances. The statistics captured in this dataset include a wide range of metrics that can be used for performance analysis, player comparison, and predictive modeling.

    Here are some key features available in the dataset:
    - **Mat**: Number of matches played by the player.
    - **Inns**: Number of innings played.
    - **NO**: Number of not outs.
    - **Runs**: Total runs scored by the player.
    - **BF**: Balls faced by the player.
    - **SR**: Strike rate of the player, calculated as `(Runs / Balls Faced) * 100`.
    - **Avg**: Batting average, calculated as `Runs / Outs` (if applicable).
    - **100**: Number of centuries scored by the player.
    - **50**: Number of half-centuries scored by the player.
    - **4s**: Number of boundaries (4s) hit by the player.
    - **6s**: Number of sixes hit by the player.

    These features provide a comprehensive view of batting performance and allow for the analysis of player consistency, strike rate, and overall impact in different match formats.

    """)

    st.markdown("### Dataset Files")
    st.write("""
    The dataset is organized into three separate files, each corresponding to a different format of cricket. Below are the details about each file:

    - **T20**: Contains batting stats for players in **T20** matches. T20 is the shortest format of cricket, with a maximum of 20 overs per team, which typically results in faster-paced games with more aggressive batting strategies.
    - **ODI**: Contains batting stats for players in **One-Day International (ODI)** matches. ODIs are 50-over matches, and the stats reflect a longer form of cricket compared to T20, with a balance between aggressive and strategic batting.
    - **Test**: Contains batting stats for players in **Test** matches. Test cricket is the longest format, lasting up to five days, and the statistics in this file represent the more patient and enduring batting performances that characterize this traditional form of cricket.

    These files contain statistics that span various cricketing eras and can be used to identify trends in player performance across different match formats and over time.

    """)

    st.markdown("### How to Use This Dataset")
    st.write("""
    This dataset can be utilized for a variety of analyses and tasks, such as:
    - **Player Comparison**: Compare batting performance across formats for different players.
    - **Trends Analysis**: Identify patterns and trends in batting performances, such as improving strike rates or increasing centuries over time.
    - **Predictive Modeling**: Use machine learning techniques to predict batting performance based on historical data (e.g., predicting a player's runs in a match based on the number of balls faced).
    - **Outlier Detection**: Spot exceptional performances or anomalies, such as a player who consistently scores high despite low balls faced.

    With this dataset, you can explore deeper insights into cricket performance, visualize trends, and apply advanced statistical techniques to extract meaningful information.

    """)

    st.markdown("### Data Quality & Potential Issues")
    st.write("""
    As with any dataset, there may be some quality issues or missing data points:
    - **Missing Values**: Certain columns might have missing or incomplete values for specific players or matches. It is important to handle missing data during analysis, either by imputing values or excluding those rows from specific analyses.
    - **Data Consistency**: Ensure consistency in column names and data formats when working with the dataset, as variations across different formats can lead to confusion in interpretation.
    - **Outliers**: Extreme performances, such as players who have played only a few matches or those who have scored very high, might introduce outliers that can affect certain analyses. Outliers should be carefully examined for potential significance.

    """)

    st.markdown("### Next Steps")
    st.write("""
    Once you've gained a better understanding of the dataset, here are some suggested next steps for deeper analysis:
    - **Data Visualization**: Create visualizations such as histograms, box plots, and scatter plots to analyze the distribution of various statistics like strike rate, runs, and centuries.
    - **Feature Engineering**: Create additional features based on existing ones, such as batting average or batting consistency.
    - **Machine Learning**: Apply machine learning algorithms to predict player performance or classify players into different performance tiers (e.g., top performers, average players, and underperformers).
    - **Player Profiles**: Use the dataset to generate detailed player profiles and track their progress over time, analyzing how their stats evolve in different formats.

    """)

    st.markdown("### Conclusion")
    st.write("""
    This cricket dataset provides a wealth of information about player performance across multiple formats. Whether you're interested in understanding cricket statistics better or applying advanced analytics techniques, this dataset serves as a great starting point for exploration.

    We hope you enjoy working with this data and uncover fascinating insights about cricket players and their performances!

    """)


