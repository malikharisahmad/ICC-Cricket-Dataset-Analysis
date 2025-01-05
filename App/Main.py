import streamlit as st
from Introduction import app as intro_app
from EDA import app as eda_app
from Statistical_Analysis import app as stats_app
from Machine_Learning import app as ml_app
from Conclusion import app as conclusion_app

PAGES = {
    "Introduction": intro_app,
    "EDA": eda_app,
    "Statistical Analysis": stats_app,
    "Machine Learning": ml_app,
    "Conclusion": conclusion_app,
}

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to:", list(PAGES.keys()))
    page = PAGES[selection]
    page()

if __name__ == "__main__":
    main()
