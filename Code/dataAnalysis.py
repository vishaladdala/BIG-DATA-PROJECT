#import the required libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

#Load the training data into pandas dataframe
trainingDataFile = sys.argv[1];
trainDf = pd.read_json(trainingDataFile)

#Plot graph for bathrooms
trainDf['bathrooms'] = trainDf['bathrooms'].map(lambda bath : 3 if bath > 3 else bath)
countFeature = trainDf['bathrooms'].value_counts()
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
ax.bar(countFeature.index.map(str), countFeature.values, alpha=0.8)
plt.xlabel('Bathrooms', fontsize=12)
plt.show()

#Plot graph for bathrooms
trainDf['bedrooms'] = trainDf['bedrooms'].map(lambda bed : 4 if bed > 4 else bed)
countFeature = trainDf['bedrooms'].value_counts()
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
ax.bar(countFeature.index.map(str), countFeature.values, alpha=0.8)
plt.xlabel('Bedrooms', fontsize=12)
plt.show()

#Plot graph for created
trainDf["created"] = pd.to_datetime(trainDf["created"])
trainDf["date_created"] = trainDf["created"].dt.date
cnt_srs = trainDf['date_created'].value_counts()
plt.figure(figsize=(14,5))
ax = plt.subplot(111)
ax.bar(cnt_srs.index, cnt_srs.values, alpha=0.8)
ax.xaxis_date()
plt.xticks(rotation='horizontal')
plt.xlabel('Created', fontsize=12)
plt.show()

trainDf["createdYear"]= trainDf["created"].dt.year
trainDf["createdMonth"]= trainDf["created"].dt.month
trainDf["createdDay"]= trainDf["created"].dt.day

#Created year
countFeature = trainDf['createdYear'].value_counts()
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
ax.bar(countFeature.index, countFeature.values, alpha=0.8)
plt.xlabel('Created Year', fontsize=12)
plt.show()

#createdMonth
countFeature = trainDf['createdMonth'].value_counts()
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
ax.bar(countFeature.index, countFeature.values, alpha=0.8)
plt.xlabel('Created Month', fontsize=12)
plt.show()

#createdDay
countFeature = trainDf['createdDay'].value_counts()
plt.figure(figsize=(6,6))
ax = plt.subplot(111)
ax.bar(countFeature.index, countFeature.values, alpha=0.8)
plt.xlabel('Created Day', fontsize=12)
plt.show()


#Num of photos
trainDf["num_photos"] = trainDf["photos"].map(len).map(lambda num : 20 if num > 20 else num)
countFeature = trainDf['num_photos'].value_counts()
plt.figure(figsize=(12,6))
ax = plt.subplot(111)
ax.bar(countFeature.index, countFeature.values, alpha=0.8)
plt.xlabel('Number of Photos', fontsize=12)
plt.show()


#num of features
trainDf["num_features"] = trainDf["features"].map(len).map(lambda num : 17 if num > 17 else num)
countFeature = trainDf['num_features'].value_counts()
plt.figure(figsize=(12,6))
ax = plt.subplot(111)
ax.bar(countFeature.index, countFeature.values, alpha=0.8)
plt.xlabel('Number of Features', fontsize=12)
plt.show()



#Interest Level
if 'interest_level' in trainDf:
	countFeature = trainDf['interest_level'].value_counts()
	plt.figure(figsize=(6,6))
	ax = plt.subplot(111)
	ax.bar(countFeature.index, countFeature.values, alpha=0.8)
	plt.xlabel('Interest Level', fontsize=12)
	plt.show()


#latitude
lowerLimit = np.percentile(trainDf.latitude.values, 1)
upperLimit = np.percentile(trainDf.latitude.values, 99)
trainDf['latitude'] = trainDf['latitude'].map(lambda lat : lowerLimit if lat < lowerLimit else upperLimit if lat > upperLimit else lat)
plt.figure(figsize=(10,8))
ax = plt.subplot(111)
ax.hist(trainDf.latitude.values, bins=50)
plt.xlabel('Latitude', fontsize=12)
plt.show()


#longitude
lowerLimit = np.percentile(trainDf.longitude.values, 1)
upperLimit = np.percentile(trainDf.longitude.values, 99)
trainDf['longitude'] = trainDf['longitude'].map(lambda lat : lowerLimit if lat < lowerLimit else upperLimit if lat > upperLimit else lat)
plt.figure(figsize=(10,8))
ax = plt.subplot(111)
ax.hist(trainDf.longitude.values, bins=50)
plt.xlabel('Longitude', fontsize=12)
plt.show()

#price
upperLimit = np.percentile(trainDf.price.values, 99)
trainDf['price'] = trainDf['price'].map(lambda lat : upperLimit if lat > upperLimit else lat)
plt.figure(figsize=(10,8))
ax = plt.subplot(111)
ax.hist(trainDf.price.values, bins=50)
plt.xlabel('price', fontsize=12)
plt.show()