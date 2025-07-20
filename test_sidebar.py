import streamlit as st

st.set_page_config(page_title="Sidebar Test", layout="centered")
st.title("Sidebar Test")

# Add sidebar content
st.sidebar.title("Test Sidebar")
st.sidebar.markdown("---")
st.sidebar.header("System Status")
st.sidebar.markdown("**Test Status:** ✅ Working")

if st.sidebar.button("Test Button"):
    st.sidebar.success("Button clicked!")
    st.success("Sidebar button works!")

st.write("If you can see the sidebar on the left, it's working correctly.")
st.write("The sidebar should contain:")
st.write("- Test Sidebar title")
st.write("- System Status section")
st.write("- Test Button")