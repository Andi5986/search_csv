import streamlit as st
import pandas as pd
import spacy
from sklearn.metrics.pairwise import cosine_similarity
from spacy.lang.en.stop_words import STOP_WORDS

# Load the Spacy model with pre-trained word vectors
nlp = spacy.load('en_core_web_md')

def main():
    uploaded_file = st.sidebar.file_uploader('Upload your Talent CSV')
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        # Concatenate all columns
        df['all_text'] = df.apply(lambda x: ' '.join(x.astype(str)), axis=1)

        # Generate word vectors for all_text column
        df['vector'] = df['all_text'].apply(lambda x: nlp(x).vector)

        st.title('Search for your star')

        search_term = st.text_input('Enter attributes')

        if search_term:
            # Ignore stop words in search terms
            search_terms = [word for word in search_term.split() if word not in STOP_WORDS]
            search_term = ' '.join(search_terms)
            search_vector = nlp(search_term).vector
            df['similarity'] = df['vector'].apply(lambda x: cosine_similarity([x], [search_vector])[0][0])

            # Generate masks for each search term
            masks = [df['all_text'].str.contains(rf"\b{term}\b", case=False, na=False, regex=True) for term in search_terms]

            # Combine the masks
            mask = masks[0]
            for m in masks[1:]:
                mask &= m

            # Only consider entries where all search terms were found
            df_filtered = df[mask]

            # Get top 100 most similar entries
            results = df_filtered.nlargest(100, 'similarity')

            st.write(results)

if __name__ == "__main__":
    main()