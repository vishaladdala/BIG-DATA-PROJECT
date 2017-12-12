// Databricks notebook source
//	https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/957490241968034/2278296410500454/8369728141520448/latest.html
/**
 * CS 6350.002 Big Data Management and Analytics Project
 * Vishal Addala  (vxa162530)
 * Pruthvi Vooka  (pxv162030)
 * Shravya Kuncha (sxk151632)
 * Likhitha Nanda (lxn160430)
**/

// COMMAND ----------

//Import all the required packages.
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.ml.classification.{LogisticRegression, LogisticRegressionModel, DecisionTreeClassifier, DecisionTreeClassificationModel, OneVsRest, OneVsRestModel, RandomForestClassifier, RandomForestClassificationModel}
import org.apache.spark.ml.feature.{StringIndexer, IndexToString, VectorAssembler, PCA, MinMaxScaler}
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.sql.functions.{lit, udf}
import org.apache.spark.ml.linalg.Vector

// COMMAND ----------

//Load the train dataset
val trainDF = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/train.csv")
//Split the train dataset into trainingData and testData in 8:2 ratio randomly.
val Array(trainingData, testData) = trainDF.randomSplit(Array(0.8, 0.2))

// COMMAND ----------

//Schema of trainDF
trainDF.printSchema()

// COMMAND ----------

/**
  * Function to  display the metrics for the PredictionAndLabel RDD.
  *
  * @param pAndL : PredictionAndLabel RDD where each element is of type (Double, Double).
  * @return Void
*/
def displayMetrics(pAndL : RDD[(Double, Double)]) {
  val metrics = new MulticlassMetrics(pAndL)
  // Confusion matrix
  println("Confusion matrix:")
  println(metrics.confusionMatrix)
  // Overall Statistics
  val accuracy = metrics.accuracy
  println("Summary Statistics")
  println(s"Accuracy = $accuracy")
  // Precision by label
  val labels = metrics.labels
  labels.foreach { l =>
    println(s"Precision($l) = " + metrics.precision(l))
  }
  // Recall by label
  labels.foreach { l =>
    println(s"Recall($l) = " + metrics.recall(l))
  }
  // False positive rate by label
  labels.foreach { l =>
    println(s"FPR($l) = " + metrics.falsePositiveRate(l))
  }
  // F-measure by label
  labels.foreach { l =>
    println(s"F1-Score($l) = " + metrics.fMeasure(l))
  }
  // Weighted stats
  println(s"Weighted precision: ${metrics.weightedPrecision}")
  println(s"Weighted recall: ${metrics.weightedRecall}")
  println(s"Weighted F1 score: ${metrics.weightedFMeasure}")
  println(s"Weighted false positive rate: ${metrics.weightedFalsePositiveRate}")
}


// COMMAND ----------

//Build common stages for pipeline

val dataFeatures = Array("bathrooms", "bedrooms", "features", "latitude", "longitude", "photos", "price", "created_month", "created_day")
//Build the String indexer for the interest level column
val interestLevelIndexer = new StringIndexer()
    .setInputCol("interest_level")
    .setOutputCol("interest_level_indexed")
    .fit(trainingData)
//Combine the features into one feature vector
val assembler = new VectorAssembler().setInputCols(dataFeatures).setOutputCol("data_features")
//Use pca to select 'k' features from the given 9 features.
val pca = new PCA()
  .setInputCol("data_features")
  .setOutputCol("pca_features")
  .setK(9)

//Convert the prediction column back to String(interest level)
val scaler = new MinMaxScaler()
  .setInputCol("pca_features")
  .setOutputCol("scaled_features")
val interestLevelStringConverter = new IndexToString().setInputCol("prediction").setOutputCol("predicted_interest_level").setLabels(interestLevelIndexer.labels)

// COMMAND ----------

//Build Logistic regression stages for pipeline
val lr = new LogisticRegression()
val ovr = new OneVsRest().setClassifier(lr)
    .setLabelCol("interest_level_indexed")
    .setFeaturesCol("scaled_features")
val lrParamGrid = new ParamGridBuilder()
    .addGrid(lr.maxIter, Array(100, 110, 90))//Int Param (>=0)
    .addGrid(lr.regParam, Array(0.0, 0.1, 0.2))//Double Param (>=0)
    .build()
val logisticRegressionStages = Array(interestLevelIndexer, assembler, pca, scaler, ovr, interestLevelStringConverter)


// COMMAND ----------

//Build the Logistic regression pipeline
val pipeline = new Pipeline().setStages(logisticRegressionStages)
val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("interest_level_indexed")
    .setPredictionCol("prediction")
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(lrParamGrid)
  .setNumFolds(10)
val lrCVModel = cv.fit(trainingData)
val logisticRegressionPredictions = lrCVModel.transform(testData)

// COMMAND ----------

//Display Logistic Regression results
val bestLRModel = lrCVModel.bestModel.asInstanceOf[PipelineModel].stages(4).asInstanceOf[OneVsRestModel].models(0).asInstanceOf[LogisticRegressionModel]
println("Max Iter: " + bestLRModel.getMaxIter)
println("Reg Param: " + bestLRModel.getRegParam)
val lrPredictionAndLabels = logisticRegressionPredictions.select("prediction", "interest_level_indexed").rdd.map{case Row(prediction: Double, interest_level_indexed: Double) => (prediction, interest_level_indexed)}
displayMetrics(lrPredictionAndLabels)

// COMMAND ----------

//Build Decision Tree stages for pipeline
val dt = new DecisionTreeClassifier()
  .setLabelCol("interest_level_indexed")
  .setFeaturesCol("scaled_features")
val dtParamGrid = new ParamGridBuilder()
  .addGrid(dt.maxBins, Array(40, 45, 50))
  .addGrid(dt.maxDepth, Array(5, 6, 8))
  .build()
val dtStages = Array(interestLevelIndexer, assembler, pca, scaler, dt, interestLevelStringConverter)

// COMMAND ----------

//Build the Decision Tree pipeline
val pipeline = new Pipeline().setStages(dtStages)
val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("interest_level_indexed")
    .setPredictionCol("prediction")
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(dtParamGrid)
  .setNumFolds(10)
val dtCVModel = cv.fit(trainingData)
val dtPredictions = dtCVModel.transform(testData)

// COMMAND ----------

//Display the Decision tree results
val bestDTModel = dtCVModel.bestModel.asInstanceOf[PipelineModel].stages(4).asInstanceOf[DecisionTreeClassificationModel]
println("Max Bins: " + bestDTModel.getMaxBins)
println("Max Depth: " + bestDTModel.getMaxDepth)
val dtPredictionAndLabels = dtPredictions.select("prediction", "interest_level_indexed").rdd.map{case Row(prediction: Double, interest_level_indexed: Double) => (prediction, interest_level_indexed)}
displayMetrics(dtPredictionAndLabels)

// COMMAND ----------

//Build Random Forest stages for pipeline
val rf = new RandomForestClassifier()
  .setLabelCol("interest_level_indexed")
  .setFeaturesCol("scaled_features")
val rfParamGrid = new ParamGridBuilder()
    .addGrid(rf.numTrees, Array(20, 30, 50))
    .addGrid(rf.maxDepth, Array(4, 6, 8))
    .build()
val rfStages = Array(interestLevelIndexer, assembler, pca, scaler, rf, interestLevelStringConverter)

// COMMAND ----------

//Build the RF pipeline
val pipeline = new Pipeline().setStages(rfStages)
val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol("interest_level_indexed")
    .setPredictionCol("prediction")
val cv = new CrossValidator()
  .setEstimator(pipeline)
  .setEvaluator(evaluator)
  .setEstimatorParamMaps(rfParamGrid)
  .setNumFolds(12)
val rfCVModel = cv.fit(trainingData)
val rfPredictions = rfCVModel.transform(testData)

// COMMAND ----------

//Display the Random Forest results
val bestRFModel = rfCVModel.bestModel.asInstanceOf[PipelineModel].stages(4).asInstanceOf[RandomForestClassificationModel]
println("Max Depth: " + bestRFModel.getMaxDepth)
println("Num Trees: " + bestRFModel.getNumTrees)
val rfPredictionAndLabels = rfPredictions.select("prediction", "interest_level_indexed").rdd.map{case Row(prediction: Double, interest_level_indexed: Double) => (prediction, interest_level_indexed)}
displayMetrics(rfPredictionAndLabels)

// COMMAND ----------

//Load the data for which we need to predict the interest level
val testDF = spark.read.format("csv").option("header", "true").option("inferSchema", "true").load("/FileStore/tables/test.csv")
//Add dummy interest_level column
val testDFWithDummy = testDF.withColumn("interest_level", lit("low": String))

// COMMAND ----------

//Run Logistic Regression on test DF
val logisticRegressionPredictions = lrCVModel.transform(testDFWithDummy)
val lrTestPredictionResults = logisticRegressionPredictions.select("listing_id", "predicted_interest_level")

// COMMAND ----------

lrTestPredictionResults.show()

// COMMAND ----------

//UDFs to divide the probabilities
val first = udf((v: Vector) => v.toArray(0))
val second = udf((v: Vector) => v.toArray(1))
val third = udf((v: Vector) => v.toArray(2))
//Labels
val labels = interestLevelIndexer.labels

// COMMAND ----------

//Run DT on testDFWithDummy
val decisionTreePredictions = dtCVModel.transform(testDFWithDummy)
val dtTestPredictionResults = decisionTreePredictions.select("listing_id", "predicted_interest_level")
val dtTestProbabilities = decisionTreePredictions.select("listing_id", "probability").withColumn(labels(0), first($"probability")).withColumn(labels(1), second($"probability")).withColumn(labels(2), third($"probability")).drop("probability")
dtTestPredictionResults.show()

// COMMAND ----------

//Write decision tree probabilities to storage
dtTestProbabilities.show()
dbutils.fs.rm("/FileStore/tables/dtTestProbabilities", true)
dtTestProbabilities.write.format("csv").option("header", "true").save("/FileStore/tables/dtTestProbabilities")

// COMMAND ----------

//Display the folder to which we wrote the decision tree probabilities
display(dbutils.fs.ls("/FileStore/tables/dtTestProbabilities"))

// COMMAND ----------

//Run RT on testDFWithDummy
val randomForestPredictions = rfCVModel.transform(testDFWithDummy)
val rfTestPredictionResults = randomForestPredictions.select("listing_id", "predicted_interest_level")
val rfTestProbabilities = randomForestPredictions.select("listing_id", "probability").withColumn(labels(0), first($"probability")).withColumn(labels(1), second($"probability")).withColumn(labels(2), third($"probability")).drop("probability")
rfTestPredictionResults.show()

// COMMAND ----------

//Write Random Forest probabilities to storage
rfTestProbabilities.show()
dbutils.fs.rm("/FileStore/tables/rfTestProbabilities", true)
rfTestProbabilities.write.format("csv").option("header", "true").save("/FileStore/tables/rfTestProbabilities")

// COMMAND ----------

//Display the folder to which we wrote the Random Forest probabilities
display(dbutils.fs.ls("/FileStore/tables/rfTestProbabilities"))
