import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



df=pd.read_csv('file1.csv')
print("Data Frame --> \n",df.head())
print("Size of Data --> ",df.shape)


import os, sys
from pyspark.sql import SparkSession

os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

spark = SparkSession.builder.master('local[*]').appName("ml_stock").getOrCreate()
dff = spark.read.csv('file1.csv', header=True)

from pyspark.sql.functions import col
nf = dff.select(*(col(c).cast('Integer').alias(c) for c in dff.columns))

from pyspark.ml.feature import VectorAssembler

vecAssembler2=VectorAssembler(inputCols=["Open", "High","Low"], outputCol="features",
   handleInvalid="keep")
nf_vecass=vecAssembler2.transform(nf)
finalized_data=nf_vecass.select("features","Close")

from pyspark.sql.functions import *
from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import TrainValidationSplit
# data = spark.read.format("libsvm")\
#     .load("data/mllib/sample_linear_regression_data.txt")
train, test = finalized_data.randomSplit([0.8, 0.2], 12)

from pyspark.ml.regression import LinearRegression
regression = LinearRegression(featuresCol='features', labelCol='Close')
regression_model = regression.fit(train)

print("**********The model has been trained***********")

regression_model.coefficients

regression_model.intercept

print("Root mean Square error on trained data --> ",regression_model.summary.rootMeanSquaredError)

print("r^2 error on trained data --> ",regression_model.summary.r2)

pred_result=regression_model.evaluate(test)

pred=pred_result.predictions.show(5)
print("**************************")
print("Root mean Square error on test data --> ",pred_result.rootMeanSquaredError)
print(" r^2 error on test data --> ",pred_result.r2)

plt.figure(figsize=(15, 6))
plt.plot([i for i in range(1,5355)],df['Close'],'Black')
plt.plot([i for i in range(1,5355)],-0.5594*df['Open'] +  0.8783*df['High'] +0.6796*df['Low'] -0.54,'red')
plt.show()
plt.savefig('stock_mark_img.png')