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
as tourism-oriented experiences since its establishment in 2008.
New York City, the most densely populated metropolis in the United States, also ranks among the globe's foremost destinations for both tourism and business endeavors.'''

'''In this project, I analyse an extensive database about variables related to Airbnb listings in the city of New York. 
The primary objective is to develop a predictive model for estimating the nightly prices of these accommodations. The project unfolds in several distinct stages:
- Firstly, an in-depth analysis of the database is conducted. This includes data cleansing procedures, involving the removal of not relevant columns and the completing of missing data points.
- Secondly, the exploration extends to the construction of insightful visualizations. Graphical representations are made to examine the behavior of different variables and assess their correlations with the target price column.
- Lastly, the dataset is split into both training and test sets. These subsets serve for the creation and assessment of regression models. 
Through these models, the aim is to derive predictive insights that contribute to the accurate estimation of the prices.'''

st.header('Dataset')

url = 'https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data'

st.write('The dataset I used can be found on the following link: [link to kaggle](' +url+')')

NY_dataset = pd.read_csv("C:\\Users\\alesi\\Desktop\\Programming\\Programming_project\\AB_NYC_2019.csv")

pd.set_option('display.float_format', '{:.3f}'.format)

st.write('The number of rows is:', NY_dataset.shape[0])
st.write('The number of columns is:', NY_dataset.shape[1])

if st.sidebar.checkbox('Have a look at the dataset'):
    st.subheader('First five rows of the dataset')
    st.write(NY_dataset.head(5))
    '''These are the variables it contains:

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

st.header('Outliers')

st.write('Histograms before removing outliers')
#Histogram before
selected_columns = st.multiselect('Select columns for histograms', ['price','minimum_nights','number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count','availability_365'])

if selected_columns:
    # Crea gli istogrammi solo per le colonne selezionate
    for column in selected_columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        NY_dataset[column].hist(ax=ax, bins = 20)
        plt.title(f'{column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot(fig)
else:
    st.warning('Please select at least one column for histograms.')

#Ouliers
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
#NY_dataset_wo.shape


st.write('Histograms after removing outliers')
selected_columns = st.multiselect('Select columns for new histograms', ['price','minimum_nights','number_of_reviews', 'reviews_per_month', 'calculated_host_listings_count','availability_365'])

if selected_columns:
    # Crea gli istogrammi solo per le colonne selezionate
    for column in selected_columns:
        fig, ax = plt.subplots(figsize=(10, 8))
        NY_dataset_wo[column].hist(ax=ax, bins = 20)
        plt.title(f'{column}')
        plt.xlabel(column)
        plt.ylabel('Frequency')
        st.pyplot(fig)
else:
    st.warning('Please select at least one column for histograms.')

if st.sidebar.checkbox('Dataset improvement'):
    st.subheader('Data before removing outliers')
    st.dataframe(NY_dataset.describe().T)

    st.subheader('Data after removing outliers')
    st.dataframe(NY_dataset_wo.describe().T)

st.write('The number of rows is:', NY_dataset_wo.shape[0])
st.write('The number of columns is:', NY_dataset_wo.shape[1])

st.header('Correlation matrix')
#Correlation matrix
st.write('If we would like too see how much the variables are correlated, we should plot the correlation matrix:')
corr = NY_dataset_wo[['price', 'minimum_nights', 'number_of_reviews','reviews_per_month', 'calculated_host_listings_count', 'availability_365']].corr()
plt.figure(figsize=(8,6))
heatmap = sns.heatmap(corr, cmap='RdBu', fmt='.3f', annot=True)
heatmap.set_xticklabels(heatmap.get_xticklabels(), fontsize=8)
heatmap.set_yticklabels(heatmap.get_yticklabels(), fontsize=8)
plt.xticks(rotation=35, ha='right')
plt.show()
st.pyplot(plt)

'''From this correlation matrix it can be seen that:
- The variables do not seem to be significantly correlated to each other;
- Only 'number_of_reviews' and 'reviews_per_month are closely correlated with each other, but is obvious;
- We can already observe that the variables are not strongly correlated with 'price' which is the target;
- Continuing with the analysis we will see how this aspect will negatively affect the models.'''

st.header('Plots')

st.subheader('Price')

plt.figure(figsize = (10,6))
NY_dataset_wo.price.value_counts().iloc[:10].plot(kind = 'bar') # The command generates a bar graph of the 10 most frequent values in the 'price' column.
st.pyplot(plt)

st.write('- More then 1750 airbnbs have a price of 150 dollars and 100 dollars each respectively.')
st.write("- Around 1350 airbnbs have circa 50 dollars price.")
st.write("- The average price is around 133 dollars.")
st.write("- 50% of data has price greater than 100 dollars")
st.write('- The most expensive airbnb has 625 dollars as price. ')

st.subheader('Minimum nights')

plt.figure(figsize = (10,6))
NY_dataset_wo.minimum_nights.value_counts().iloc[:10].plot(kind = 'bar')
st.pyplot(plt)

st.write('Almost 10k people stay 1 night in airbnb and 3.5k have choosen to stay a month.')

st.subheader('Number of reviews')

plt.figure(figsize = (10,6))
NY_dataset_wo.number_of_reviews.value_counts().iloc[:10].plot(kind = 'bar')
st.pyplot(plt)

st.write('We can see that many airbnbs do not have reviews.')

st.subheader('Availability')

value_counts = NY_dataset_wo.availability_365.value_counts()
value_0 = value_counts.iloc[0] # I take the value 0 because is the max
other_values_sum = sum(value_counts.values) - value_0 # take the sum of all the other values

values = [value_0, other_values_sum]
labels = [f'Value 0 ({value_0})', f'Other Values ({other_values_sum})']
plt.figure(figsize=(10, 6))
plt.pie(values, labels=labels, autopct='%1.1f%%', colors=['blue', 'gray'])
plt.axis('equal')
plt.title("0 days and all the others")
st.pyplot(plt)

st.write('It is instersting that 40.3% of all airbnbs have 0 days availability.') 
'''This data might seem unusual, but upon further investigation, it can be understood that Airbnb accommodations can also be booked for extended periods, likely for work or study purposes, and are paid as monthly rents.'''

st.header('Neighborhood group')

# Calculating the number of Airbnbs for each boroughs
neighbourhood = NY_dataset_wo.neighbourhood_group.value_counts()

# Plotting the pie plot of room type
plt.figure(figsize=(10, 6))
plt.pie(neighbourhood, labels = neighbourhood.index, colors = ['darkorange', 'blue', 'green', 'darkred', 'purple'], autopct='%1.1f%%', startangle = 140)
plt.axis('equal')
plt.title('Boroughs')
st.pyplot(plt)

st.write('Amount of airbns per boroughs:')
'''- Manhattan: 18,085'''
'''- Brooklyn: 18,084'''
'''- Queens: 4,862'''
'''- Bronx: 955'''
'''- Staten Island: 321'''

st.header('Room type')

# Calculating the number of rooms for each room type
room_type = NY_dataset_wo.room_type.value_counts()

# Plotting the pie plot of room type
plt.figure(figsize=(10, 6))
plt.pie(room_type, labels = room_type.index, colors = ['yellow', 'darkcyan', 'pink'], autopct='%1.1f%%', startangle = 140)
plt.axis('equal')
plt.title('Room type')
st.pyplot(plt)

st.write('Amount of airbns per room type:')
'''- Entire home/apt: 21,365'''
'''- Private room: 19,878'''
'''- Shared room: 1,064'''

st.header('Average price per room type')

# Calculating the average price per room type
type_price = NY_dataset_wo.groupby('room_type').price.mean().sort_values(ascending=False)

# Plottingthe average price per room type
plt.figure(figsize=(10, 6))
ax = sns.barplot(x = type_price.index, y = type_price, palette = ['yellow', 'darkcyan', 'pink']) 
ax.set_title('Average Price per Room Type')
ax.tick_params(bottom=False, top=False, left=False, right=False)
ax.set_ylabel('$', fontsize=12)
st.pyplot(plt)

st.write('Average price per room type:')
'''- Entire home/apt: 182.843'''
'''- Private room: 82.826'''
'''- Shared room: 64.235'''

st.header('Average price per borough')

# Calculating the average price per borough
price_boroughs = NY_dataset_wo.groupby('neighbourhood_group').price.mean().sort_values(ascending=False)

# Plotting the average price per borough
plt.figure(figsize =(10,6))
ax = sns.barplot(x = price_boroughs.index, y = price_boroughs, palette=['darkorange', 'blue', 'green', 'purple', 'darkred', ]) 
ax.set_title('Average Price per Borough')
ax.tick_params(bottom=False, top=False, left=False, right=False)
ax.set_ylabel('$', fontsize=12)
st.pyplot(plt)

st.write('Average price per borough:')
'''- Manhattan: 164.706'''
'''- Brooklyn: 114.301'''
'''- Queens: 95.724'''
'''- Staten Island: 94.801'''
'''- Bronx: 83.403'''

st.header('Price and type of room for borought')

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=NY_dataset_wo.neighbourhood_group, y=NY_dataset_wo.price, hue=NY_dataset_wo.room_type, errorbar=None, edgecolor='white', palette = ['blue', 'darkorange', 'green'] )
ax.set_title('Price and type of room for borought')
st.pyplot(plt)

'''As we can see, Manhattan is the most expensive neighbourhood and the price of entire home/apt is more than any other room type.'''

st.header('Room type location')
# Lista dei room_type disponibili nel tuo dataset
available_room_types = NY_dataset['room_type'].unique()

# Qui inizia la parte del codice relativa ai checkbox
st.sidebar.subheader('Select Room Types')
selected_room_types = st.sidebar.multiselect('Select room types to show', available_room_types, default=available_room_types)

# Creazione del grafico scatterplot
title = 'Room type location per Neighbourhood Group'
plt.figure(figsize=(8, 6))

# Filtra il DataFrame in base ai room_type selezionati
filtered_df = NY_dataset[NY_dataset['room_type'].isin(selected_room_types)]

# Creazione del grafico scatterplot effettivo
sns.scatterplot(data=filtered_df, x='longitude', y='latitude', hue='room_type', palette='bright', alpha=0.5).set_title(title)

# Visualizza il grafico
st.pyplot(plt)

st.header('Neighbourhood Group Location')

plt.figure(figsize=(10, 6))
ax = sns.scatterplot(x=NY_dataset_wo.longitude,y=NY_dataset_wo.latitude, hue=NY_dataset_wo.neighbourhood_group, palette = 'bright')
ax.set_title('Neighbourhood Group Location')
st.pyplot(plt)

'''With all these plots, the conclusions are:'''
'''- Entire home/apt and private room are the most common room types;'''
'''- Entire home/apt are usually more expensive than private and shared rooms;'''
'''- Over 85% of the rooms are located in Manhattan and Brooklyn, which are also the most expensive regions, especially Manhattan.'''

if st.sidebar.checkbox('Last phase of data preparation'):
    st.subheader('The code of the passages')
    with st.echo():
        NY_final = NY_dataset_wo.drop(['id','host_id','latitude','longitude','neighbourhood'], axis = 1, inplace = False)
        x = NY_final.iloc[:,[0,1,3,4,5,6,7]]
        y = NY_final['price']
        #I dropped these columns as they are not useful for the analysis. 
        #Specifically, knowing the property's ID and owner's ID might be essential if I were to create a database, but not for analyzing the Airbnb price per night.
        #As for the other columns, I consider them to be too specific, and the 'neighborhood_group' (218 different neighbothood) column, in my opinion, is adequate for the analysis.

        #One-Hot Encoding
        X = pd.get_dummies(NY_final.iloc[:,[0,1,3,4,5,6,7]], prefix=['neighbourhood_group', 'room_type'], drop_first=True) 
        #Drop_first = True; indicates that one of the categories of each categorical variable must be left out to avoid the so-called "dummy variable trap". 
        #Dummy variable trap occurs when dummy variables are highly correlated.
        

        #Splitting the dataset into test and training data
        X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.2, random_state= 42)

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.fit_transform(X_test)

        X_low = X.loc[(NY_dataset_wo['price'] < 249)]
        y_low = y.loc[(NY_final['price'] < 249)]

        #Splitting the dataset into test and training data
        X_train_low, X_test_low, y_train_low, y_test_low =  train_test_split(X_low, y_low,test_size = 0.2, random_state= 42)

    st.write('I encoded the values of the ''room_type'' and ''neighborhood_group'' columns because they contain categorical data, rather than numerical data.') 
    st.write('To achieve this, I utilized the get_dummies method from the Pandas library.')
    st.write('Get_dummies is a method that convert categorical variable into dummy/indicator variables.') 
    st.write('Each variable is converted in as many 0/1 variables as there are different values. Columns in the output are each named after a value.')
    st.write('You have to pay attention when you are encoding such data, as it is not always possible to assign values from 0 to n, where n represents the number of different categories in the column.')
    st.write('To achieve this, there needs to be some form of ordinal relationship among the data, which is not present in this case.') 
pass

NY_final = NY_dataset_wo.drop(['id','host_id','latitude','longitude','neighbourhood'], axis = 1, inplace = False)
x = NY_final.iloc[:,[0,1,3,4,5,6,7]]
y = NY_final['price']

X = pd.get_dummies(NY_final.iloc[:,[0,1,3,4,5,6,7]], prefix=['neighbourhood_group', 'room_type'], drop_first=True) 
        
X_train, X_test, y_train, y_test =  train_test_split(X,y,test_size = 0.2, random_state= 42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

X_low = X.loc[(NY_dataset_wo['price'] < 249)]
y_low = y.loc[(NY_final['price'] < 249)]

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

            plt.figure(figsize=(7,4))
            sns.regplot(y=y_pred, x=y_test, line_kws={"color": "red"}, color='blue')
            plt.title('Evaluated predictions', fontsize=15)
            plt.xlabel('Predicted values')
            plt.ylabel('Real values')
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

            plt.figure(figsize=(7,4))
            sns.regplot(y=y_pred_low, x=y_test_low, line_kws={"color": "red"}, color='blue')
            plt.title('Evaluated predictions', fontsize=15)
            plt.xlabel('Predicted values')
            plt.ylabel('Real values')
            st.pyplot(plt)
pass

'''The properties in the dataset exhibit substantial variations in prices.'''

'''Dividing the dataset into different price categories proves valuable for analysis purposes.'''

'''The most significant variables for predicting prices are:'''
'''- Neighborhood_group'''
'''- Type_of_room'''

'''The models aimed at predicting prices have shown not noticeable performance.'''
'''- The highest achieved model score is 0.52.'''

'''Predictions tend to be more accurate for properties priced under $249, which constitutes around 90% of the dataset.'''

'''The dataset suffers from inadequate data quality, with a notable challenge being the uneven distribution of features. This has posed difficulties in constructing an effective predictive model.'''

'''Exploring deeper into the underlying problem and augmenting the dataset with additional valuable features that demonstrate stronger correlations with the target variable could potentially lead to improved price predictions.'''


