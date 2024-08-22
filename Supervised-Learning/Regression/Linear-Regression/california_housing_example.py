#based on https://inria.github.io/scikit-learn-mooc/python_scripts/datasets_california_housing.html
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error, r2_score
from sklearn import model_selection as  model_selection
#load the dataset
from sklearn.datasets import fetch_california_housing
import pandas as pd

# to print the attributes of the dataset

california_housing = fetch_california_housing(as_frame=True)

#The dependent variable is median house value
#Number of Instances: 20640

#Number of Attributes: 8 numeric

#Attribute Information:
#       - MedInc        median income in block group
#       - HouseAge      median house age in block group
#       - AveRooms      average number of rooms per household
#       - AveBedrms     average number of bedrooms per household
#       - Population    block group population
#       - AveOccup      average number of household members
#       - Latitude      block group latitude
#       - Longitude     block group longitude
