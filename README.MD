# Talent Search Application

This is a talent search application which allows users to search a CSV of talent attributes using natural language.

## Installation

1. Clone this repository:
    ```
    git clone https://github.com/yourusername/talent-search.git
    cd talent-search
    ```

2. Install the required packages:
    ```
    pip install -r requirements.txt
    ```

## Usage

1. Run the Streamlit application:
    ```
    streamlit run app.py
    ```

2. Navigate to the URL provided in your console (usually `http://localhost:8501`).

3. Use the sidebar to upload your Talent CSV.

4. Enter your search terms into the search bar and press Enter to conduct a search.

## Features

- Natural language search across all columns of the CSV
- Relevance ranking of search results
- Ignore common English stop words (like 'with', 'from', etc.) in search terms
