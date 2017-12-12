-------------------------------------------------------------------------------------------
 CS 6350.002 Big Data Management and Analytics Project
-------------------------------------------------------------------------------------------
Team
* Vishal Addala  (vxa162530)
* Pruthvi Vooka  (pxv162030)
* Shravya Kuncha (sxk151632)
* Likhitha Nanda (lxn160430)
-------------------------------------------------------------------------------------------
-------------------------------------------------------------------------------------------
The databricks public link for the project is

	https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/957490241968034/2278296410500454/8369728141520448/latest.html

The kaggle competition link is

	https://www.kaggle.com/c/two-sigma-connect-rental-listing-inquiries

The data analysis and preprocessing is done in Python using the following packages

	1. numpy
	2. matplotlib.pyplot
	3. pandas

They can be installed using pip if not already there.

To get the distribution charts for the data, use the following commands

	$ python dataAnalysis.py train.json

	$ python dataAnalysis.py test.json

To preprocess and get the csv files use the following commands,

	$ python preprocess.py train.json train.csv

	$ python preprocess.py test.json test.csv

The data for the project is in the Data Folder. They are train.csv and test.csv

The packages used in Spark are

	1. org.apache.spark.mllib
	2. org.apache.spark.ml

The classifiers used are LogisticRegression, DecisionTreeClassifier, RandomForestClassifier.

Run the file BDMA Project.scala to run the classifiers or import the notebook from the link given in the starting of this readme file.