import streamlit as st
from pymongo import MongoClient

# Connection
client = MongoClient("mongodb://localhost:27017/")
db = client["MRBOFER"]  # Database name
collection = db["Stress"]  # Collection name

# Fetch
def view_all_data():
    data = list(collection.find())  # Fetch all documents in the collection
    return data
