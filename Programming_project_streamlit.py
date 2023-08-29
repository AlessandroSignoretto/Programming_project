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

st.subheader('A short explanation of the project topic')

'''**Airbnb** serves as an internet platform that facilitates the organization and provision of accommodations, predominantly homestays, as well 
as tourism-oriented experiences since its establishment in 2008. <br>
New York City, the most densely populated metropolis in the United States, also ranks among the globe's foremost destinations for both tourism and business endeavors.'''

url = 'https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data'

st.write('The dataset I used can be found on the following link: [link to kaggle](' +url+')')

NY_dataset = pd.read_csv("C:\\Users\\alesi\\Desktop\\Programming\\Programming_project\\AB_NYC_2019.csv")

if st.sidebar.checkbox('Have a look at the dataset'):
    st.write('First five rows of the dataset')
    st.write(NY_dataset.head(5))
    '''Before we perform any analysis, we'll first see what our dataset looks like. These are the variables it contains:

    - id: - id number that identifies the property
    - name: - Property name
    - host_id: - id number that identifies the host
    - host_name: - Host name
    - neighbourhood_group: - The main regions of the city
    - neighbourhood: - The neighbourhoods
    - latitude: - Property latitude
    - longitude: - Property longitude
    - room_type: - Type of the room
    - price: - The price for one night
    - minimum_nights: - Minimum amount of nights to book the place
    - number_of_reviews: - Number of reviews received
    - last_review: - Date of the last review
    - reviews_per_month: - Amount of reviews per month
    - calculated_host_listings_count: - Number of properties available on Airbnb owned by the host
    - availability_365: - Number of days of availability within 365 days'''

#Data cleaning
NY_dataset.drop(['name', 'host_name', 'last_review'], axis=1, inplace=True)
NY_dataset.fillna({'reviews_per_month': 0}, inplace=True)

#Outliers
# Select columns
NY_dataset_col = NY_dataset[['price', 'minimum_nights','number_of_reviews', 'reviews_per_month','calculated_host_listings_count' ]]

for col in NY_dataset_col:
    # Calculate z-score of the columns
    z_score = np.abs(stats.zscore(NY_dataset[col]))
    outliers_num = len(np.where(z_score > 2)[0])
    if outliers_num:
        print('{}: {}'.format(col, outliers_num))

# Remove outliers outside 2 standard deviations from the mean
z_scores = np.abs(stats.zscore(NY_dataset_col))

# DataFrame without outliers
NY_dataset_wo = NY_dataset[(z_scores < 2).all(axis=1)]
NY_dataset_wo.shape


#Correlation matrix
st.write('If we would like too see how much the variables are correlated, we should plot the correlation matrix:')
corr = NY_dataset_wo[['price', 'minimum_nights', 'number_of_reviews','reviews_per_month', 'calculated_host_listings_count', 'availability_365']].corr()
plt.figure(figsize=(5,4))
heatmap = sns.heatmap(corr, cmap='RdBu', fmt='.3f', annot=True)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=8)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=8)
plt.xticks(rotation=35, ha='right')
plt.show()
st.pyplot(plt)

#Analysis
#Regression analysis to predict the price
NY_final = NY_dataset_wo.drop(['id','host_id','latitude','longitude','neighbourhood'], axis = 1, inplace = False)
x = NY_final.iloc[:,[0,1,3,4,5,6,7]]
y = NY_final['price']

#One-Hot Encoding
X = pd.get_dummies(NY_final.iloc[:,[0,1,3,4,5,6,7]], prefix=['neighbourhood_group', 'room_type'], drop_first=True) 
#Drop_first = True; indicates that one of the categories of each categorical variable must be left out to avoid the so-called "dummy variable trap". 
#Dummy variable trap occurs when dummy variables are highly correlated.

#Splitting the dataset into test and training data
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.2, random_state= 42)




#GRAFICO NEW YORK
# Lista dei room_type disponibili nel tuo dataset
available_room_types = NY_dataset['room_type'].unique()

# Qui inizia la parte del codice relativa ai checkbox
st.sidebar.subheader('Select Room Types')
selected_room_types = st.sidebar.multiselect('Select room types to show', available_room_types, default=available_room_types)

# Creazione del grafico scatterplot
title = 'Room type location per Neighbourhood Group'
plt.figure(figsize=(7, 4))

# Filtra il DataFrame in base ai room_type selezionati
filtered_df = NY_dataset[NY_dataset['room_type'].isin(selected_room_types)]

# Creazione del grafico scatterplot effettivo
sns.scatterplot(data=filtered_df, x='longitude', y='latitude', hue='room_type', palette='bright', alpha=0.5).set_title(title)

# Visualizza il grafico
st.pyplot(plt)






scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

X_low = X.loc[(NY_dataset_wo['price'] < 249)]
y_low = y.loc[(NY_final['price'] < 249)]

#Splitting the dataset into test and training data
X_train_low, X_test_low, y_train_low, y_test_low =  train_test_split(X_low, y_low,test_size = 0.2, random_state= 42)


with st.expander('Models to predict the price'):

    st.subheader('$R^{2}$, $RMSE$ $and$ $plot$ $error$')

    select_price = st.selectbox('Select price:', ['All', 'Price under $249'])
    if select_price == 'All':
        with st.spinner('Training...'):
            select_model = st.selectbox('Select model:', ['Linear Regression','Ridge','Regression Tree','Random Forest', 'Gradient Boosting'])
            
            if select_model == 'Linear Regression':
                model = LinearRegression()
            elif select_model == 'Ridge':
                model = Ridge(alpha=0.01)
            elif select_model == 'Regression Tree':  
                model = DecisionTreeRegressor(min_samples_leaf=.01)
            elif select_model == 'Random Forest':
                model = RandomForestRegressor(n_estimators=200, max_depth = 45, min_samples_leaf = 20)
            elif select_model == 'Gradient Boosting':  
                model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01)
            
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            st.write("R2 score: ", r2_score(y_test, y_pred))
            st.write("RMSE: ", np.sqrt(mean_squared_error(y_test, y_pred)))

            error_diff = pd.DataFrame({'Actual Values': np.array(y_test).flatten(), 'Predicted Values': y_pred.flatten()})

            df1 = error_diff.head(20)
            df1.plot(kind='bar', figsize=(7, 4))
            plt.grid(linewidth='0.5', color='black')
            st.pyplot(plt)
            pass

    elif select_price == 'Price under $249':
        with st.spinner('Training...'):
            select_model = st.selectbox('Select model:', ['Linear Regression','Ridge','Regression Tree','Random Forest', 'Gradient Boosting'])
            
            if select_model == 'Linear Regression':
                model = LinearRegression()
            elif select_model == 'Ridge':
                model = Ridge(alpha=0.01)
            elif select_model == 'Regression Tree':  
                model = DecisionTreeRegressor(min_samples_leaf=.01)
            elif select_model == 'Random Forest':
                model = RandomForestRegressor(n_estimators=200, max_depth = 45, min_samples_leaf = 20)
            elif select_model == 'Gradient Boosting':  
                model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.01)
            
            model.fit(X_train_low,y_train_low)
            y_pred_low=model.predict(X_test_low)

            #R2 score
            st.write("R2 score: ",r2_score(y_test_low,y_pred_low))
            st.write("RMSE: ",np.sqrt(mean_squared_error(y_test_low,y_pred_low)))

            #Error
            error_diff = pd.DataFrame({'Actual Values': np.array(y_test_low).flatten(), 'Predicted Values': y_pred_low.flatten()})

            #Visualize
            df1 = error_diff.head(20)
            df1.plot(kind='bar',figsize=(7,4))
            plt.grid(linewidth = '0.5', color = 'black')
            st.pyplot(plt)
        


