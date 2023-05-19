import streamlit as st
import pandas as pd

uploaded_file = st.sidebar.file_uploader('Upload your Talent CSV')
if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

def main():
    st.title('Search for your star')

    search_term = st.text_input('Enter attributes')

    if search_term:
        # Split the search term into individual words
        search_terms = search_term.split()

        # Initialize a mask to select all rows
        mask = pd.Series([True] * len(df))

        for term in search_terms:
            # Case-insensitive search in all String columns
            string_cols = df.select_dtypes(include='object')
            term_mask = string_cols.apply(lambda x: x.str.contains(term, case=False, na=False)).any(axis=1)

            # Case-insensitive search in all numeric columns
            numeric_cols = df.select_dtypes(include='number')
            term_mask |= numeric_cols.apply(lambda x: x == pd.to_numeric(term, errors='coerce')).any(axis=1)
            
            # Combine the mask for this term with the overall mask
            mask &= term_mask

        results = df[mask]
        
        st.write(results)

if __name__ == "__main__":
    main()