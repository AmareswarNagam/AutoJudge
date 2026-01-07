# importing required dependencies
import streamlit as st
import numpy as np
import re
import joblib

# loading saved models
vectorizer = joblib.load("vectorizer.joblib")
classifier = joblib.load("classifier.joblib")
regressor = joblib.load("regressor.joblib")



