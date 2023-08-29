import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

st.title('Programming project')

st.header('NYC Airbnb: Price prediction')

st.subheader('A short explanation of the project')

st.write("Airbnb serves as an internet platform that facilitates the organization and provision of accommodations, predominantly homestays, as well as tourism-oriented experiences since its establishment in 2008. New York City, the most densely populated metropolis in the United States, also ranks among the globe's foremost destinations for both tourism and business endeavors.")

NY_dataset = pd.read_csv("C:\\Users\\alesi\\Desktop\\Programming\\Programming_project\\AB_NYC_2019.csv")

st.write(NY_dataset.head(5))