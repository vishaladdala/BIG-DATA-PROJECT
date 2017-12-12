import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys


trainingDataFile = sys.argv[1];
trainDf = pd.read_json(trainingDataFile)

trainDf['bathrooms'] = trainDf['bathrooms'].map(lambda bath : 3 if bath > 3 else bath)
trainDf['bedrooms'] = trainDf['bedrooms'].map(lambda bed : 4 if bed > 4 else bed)
trainDf["created"] = pd.to_datetime(trainDf["created"])
trainDf["created_month"]= trainDf["created"].dt.month
trainDf["created_day"]= trainDf["created"].dt.day
trainDf["photos"] = trainDf["photos"].map(len).map(lambda num : 20 if num > 20 else num)
trainDf["features"] = trainDf["features"].map(len).map(lambda num : 17 if num > 17 else num)
lowerLimit = np.percentile(trainDf.latitude.values, 1)
upperLimit = np.percentile(trainDf.latitude.values, 99)
trainDf['latitude'] = trainDf['latitude'].map(lambda lat : lowerLimit if lat < lowerLimit else upperLimit if lat > upperLimit else lat)
lowerLimit = np.percentile(trainDf.longitude.values, 1)
upperLimit = np.percentile(trainDf.longitude.values, 99)
trainDf['longitude'] = trainDf['longitude'].map(lambda lat : lowerLimit if lat < lowerLimit else upperLimit if lat > upperLimit else lat)
upperLimit = np.percentile(trainDf.price.values, 99)
trainDf['price'] = trainDf['price'].map(lambda lat : upperLimit if lat > upperLimit else lat)
trainDf.drop(["description", "street_address", "created", "display_address", "building_id", "manager_id"], axis = 1, inplace = True)
trainDf.to_csv(sys.argv[2])