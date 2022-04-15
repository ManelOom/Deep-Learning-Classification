import tensorflow as tf
from tensorflow import keras
import pandas as pd

# Firstly, we'll import our working Dataset. The WHO's Global Health Observatory (GHO) data repository tracks life expectancy for countries worldwide 
# by following health status and many other related factors.

# Although there have been a lot of studies undertaken in the past on factors affecting life expectancy considering demographic variables, income composition,
# and mortality rates, it was found that the effects of immunization and human development index were not taken into account.

# This dataset covers a variety of indicators for all countries from 2000 to 2015 including:
# immunization factors
# mortality factors
# economic factors
# social factors
# other health-related factors

# Ideally, this data will eventually inform countries concerning which factors to change in order to improve the life expectancy of their populations.
# If we can predict life expectancy well given all the factors, this is a good sign that there are some important patterns in the data.
# Life expectancy is expressed in years, and hence it is a number. This means that in order to build a predictive model one needs to use regression.

#1. Data loading and observing

dataset  = pd.read_csv('life_expectancy.csv')
print(dataset.head())
print(dataset.describe())
print(dataset.shape)


check_duplicates = dataset.duplicated().sum()
check_non_duplicates = ~dataset.duplicated()
print(check_duplicates)
print(check_non_duplicates.sum())
#No duplicates, we'll assume this data is clean since it was previously treated by its source.

# Since knowing where the data comes from might be confusing when developing a predictive
# model and since the column "Country" can't be generalized over, we'll drop this column.

dataset = dataset.drop(labels = 'Country', axis=1)

#Create features and labels

labels = dataset.iloc[:,-1]
features = dataset.iloc[:,0:-1]

print("Labels here")
print(labels)
print("------------")
print("Features here")
print(features)

#2. Data Preprocessing

# It's impossible to deal with categorical variables when leading with deep learning models,
# so we must convert our categorical variables to numerical.
# We'll do this through one-hot-encoding in this case.

features = pd.get_dummies(features)
# We'll verify if it was correctly done.
pd.set_option('display.max_columns', None)
print(features)
# As we can see, it was correctly done

# Now, we'll split our data into training set and test set.

from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size= 0.2, random_state = 42)

# Now we must normalize our numerical features. 

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import Normalizer
from sklearn.compose import ColumnTransformer

# Variable to store our numerical features.
numerical_features = features.select_dtypes(include=['float64', 'int64'])
# Variable to store our numerical columns.
numerical_columns = numerical_features.columns
print("These are our numerical columns:")
print("                          ")
print(numerical_columns)

# Finally, we'll use ColumnTransformer to normalize our numeric features.

Col_Transform = ColumnTransformer([("only numeric", StandardScaler(), numerical_columns)], remainder='passthrough')

# Fit Col_Transform to the training data

features_train_scaled = Col_Transform.fit_transform(features_train)

# Fit Col_Transform to the test data

features_test_scaled = Col_Transform.fit_transform(features_test)


#3. Building the Model

from tensorflow.keras.models import Sequential

Model_DL = Sequential()

# Create input layer to the network model
# Shape corresponding to the number of features in dataset

from tensorflow.keras.layers import InputLayer

Input_DL = InputLayer(input_shape = (dataset.shape[1], ))

# Add layer to the model

Model_DL.add(Input_DL)

# Add one Dense hidden layer with some hidden units, using the relu activation function

from tensorflow.keras.layers import Dense

Model_DL.add(Dense(64, activation = "relu"))

# Add a Dense output layer with only one neuron, because we need a single output for a regression prediction

from tensorflow.keras.layers import Dense

Model_DL.add(Dense(1))

# Model Summary

print(Model_DL.summary())


#4. Initializing the optimizer and compiling the model

# We'll be using the "Adam" optimizer, with a learning rate of 0.01

from tensorflow.keras.optimizers import Adam

Optimize_Model_DL = Adam(learning_rate = 0.01)

# Now that our optimizer is initiallized, we'll compile the model using
# the Sequential.compile() method.

Model_DL.compile(loss = 'mse', metrics = ['mae'], optimizer = Optimize_Model_DL)

#5. Fit and evaluate the model

# We'll train our model with the Sequential.fit() method 

Model_DL.fit(features_train_scaled, labels_train, epochs = 50, batch_size = 1, verbose = 1)

# We'll use the Sequential.evaluate() method to evaluate our trained model
# on the preprocessed test data set, and with the test labels.

Final_mse, Final_mae = Model_DL.evaluate(features_test_scaled, labels_test, verbose = 0)

# Lets see our final loss (RMSE) and final metric (MAE) to check our models predictive performance on the test set.

print(Final_mse, Final_mae)

# We end up getting a RMSE (our loss function, since we're handling a Regression problem) of around 7
# and an average Mean Square Error of aproximately 2. 
# Therefore, the model ends up having a satisfatory performance overall.