Project 
Title: CSV Searcher

A simple web application using Streamlit and Pandas, to help you search through a database of talented individuals based on specific attributes.

Table of Contents

Installation
Usage
License
Installation


Before starting, ensure you have the following requirements:

Python 3.6 or later

pandas==2.0.1
streamlit==1.22.0

You can install these packages using pip:

pip install streamlit pandas

Start the Streamlit app by navigating to the directory containing the script and running:
bash

streamlit run main.py


Navigate to the URL provided in your terminal to view the app in your web browser.


On the sidebar, upload your CSV file containing data about various talents using the 'Upload your Talent CSV' button.


In the main section of the app, input any attributes you wish to search for in the database.


The application will display the corresponding rows from the CSV file that match your search criteria. The search is case-insensitive and can be performed on both string and numeric columns.

License

This project is licensed under the MIT License. See the LICENSE file for details.