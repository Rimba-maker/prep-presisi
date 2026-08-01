import streamlit as st

st.set_page_config(
    page_title="Prep Presisi", page_icon=":material/storefront:", layout="wide"
)

page = st.navigation(
    [
        st.Page("overview_page.py", title="Overview", icon=":material/dashboard:"),
        st.Page("detail_page.py", title="Detail outlet", icon=":material/storefront:"),
    ]
)
page.run()
