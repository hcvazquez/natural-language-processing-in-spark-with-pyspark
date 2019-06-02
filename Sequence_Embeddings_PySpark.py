# Databricks notebook source
import findspark
findspark.init()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.sql.functions import *

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.sql.window import Window

# scikit-learn
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from statsmodels.api import Logit
from sklearn.decomposition import PCA
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
%matplotlib inline

# others
import pandas as pd
import numpy as np
import sys
import itertools
import re
from random import sample
import time

# COMMAND ----------

!pip install gensim

# COMMAND ----------

from gensim.models.doc2vec import LabeledSentence
from gensim.models import Doc2Vec
from gensim.models import Word2Vec

# COMMAND ----------

#create SparkSession
spark=SparkSession.builder.appName('seq_embedding').getOrCreate()

# COMMAND ----------

#reading a file
df = spark.read.csv('embedding_dataset.csv',header=True,inferSchema=True)

# COMMAND ----------

df.count()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

df.select('user_id').distinct().count()

# COMMAND ----------

df.groupBy('page').count().orderBy('count',ascending=False).show(10,False)

# COMMAND ----------

df.select(['user_id','page','visit_number','time_spent','converted']).show(10,False)

# COMMAND ----------

# window for each user order by timestamp
w = Window.partitionBy("user_id").orderBy('timestamp')

# COMMAND ----------

#creating a lagged column 
df = df.withColumn("previous_page", lag("page", 1, 'started').over(w))

# COMMAND ----------

df.select('user_id','timestamp','previous_page','page').show(10,False)

# COMMAND ----------

# adding an indicator if current page is same as next page
def indicator(page, prev_page):
    if page == prev_page:
        return 0
    else:
        return 1
    
page_udf = udf(indicator,IntegerType())

# COMMAND ----------

# adding a column for indicator and cumulative indicator
df = df.withColumn("indicator",page_udf(col('page'),col('previous_page'))) \
        .withColumn('indicator_cummulative',sum(col('indicator')).over(w))

# COMMAND ----------

df.select('previous_page','page','indicator','indicator_cummulative').show(20,False)

# COMMAND ----------

# create window with user and indicator cummulative
w2 = Window.partitionBy(["user_id",'indicator_cummulative']).orderBy('timestamp')

# COMMAND ----------

# adding a column with time spent cumulative ( time spent by a user on a page  visited in continuation )
df = df.withColumn('time_spent_cummulative',sum(col('time_spent')).over(w2))

# COMMAND ----------

df.select('timestamp','previous_page','page','indicator','indicator_cummulative','time_spent','time_spent_cummulative').show(20,False)

# COMMAND ----------

# creating a window to get final page and final timespent 
w3 = Window.partitionBy(["user_id",'indicator_cummulative']).orderBy(col('timestamp').desc())

# COMMAND ----------

# Add column for final page category and final time spent
df = df.withColumn('final_page',first('page').over(w3))\
     .withColumn('final_time_spent',first('time_spent_cummulative').over(w3))


# COMMAND ----------

df.select(['time_spent_cummulative','indicator_cummulative','page','final_page','final_time_spent']).show(10,False)

# COMMAND ----------

# user and pagelevel aggregation  
aggregations=[]
aggregations.append(max(col('final_page')).alias('page_emb'))
aggregations.append(max(col('final_time_spent')).alias('time_spent_emb'))
aggregations.append(max(col('converted')).alias('converted_emb'))

# COMMAND ----------

#selecting relevant columns
# extracting the dataframe with the data frame that will be used for embedding
df_embedding = df.select(['user_id','indicator_cummulative','final_page','final_time_spent','converted']).groupBy(['user_id','indicator_cummulative']).agg(*aggregations)

# COMMAND ----------

df_embedding.count()

# COMMAND ----------

df_embedding.show(30, False)

# COMMAND ----------

# create a partition by user id ordered by indicator cumulative to get the journey
w4 = Window.partitionBy(["user_id"]).orderBy('indicator_cummulative')
w5 = Window.partitionBy(["user_id"]).orderBy(col('indicator_cummulative').desc())

# COMMAND ----------

df_embedding = df_embedding.withColumn('journey_page', collect_list(col('page_emb')).over(w4))\
                          .withColumn('journey_time_temp', collect_list(col('time_spent_emb')).over(w4)) \
                         .withColumn('journey_page_final',first('journey_page').over(w5))\
                        .withColumn('journey_time_final',first('journey_time_temp').over(w5)) \
                        .select(['user_id','journey_page_final','journey_time_final','converted_emb'])

# COMMAND ----------

df_embedding.select('user_id','journey_page_final','journey_time_final').show(10)

# COMMAND ----------

df_embedding.count()

# COMMAND ----------

df_embedding.select('user_id').distinct().count()

# COMMAND ----------

df_embedding = df_embedding.dropDuplicates()

# COMMAND ----------

df_embedding.count()

# COMMAND ----------

df_embedding.select('user_id').distinct().count()

# COMMAND ----------

df_embedding.select('user_id','journey_page_final','journey_time_final').show(10)

# COMMAND ----------

df_embedding.explain(true) 

# COMMAND ----------

# create pandas dataframe for embedding
pd_df_embedding = df_embedding.toPandas()

# COMMAND ----------

pd_df_embedding.head(5)

# COMMAND ----------

# making sure we don't have journeys with length less than 4
pd_df_embedding = pd_df_embedding[pd_df_embedding['journey_length'] > 4 ]

# COMMAND ----------

# reset index
pd_df_embedding = pd_df_embedding.reset_index(drop=True)

# COMMAND ----------

# train model
EMBEDDING_SIZE = 100
model = Word2Vec(pd_df_embedding['journey_page_final'], size=EMBEDDING_SIZE)

# COMMAND ----------

model.total_train_time

# COMMAND ----------

# summarize the loaded model
print(model)

# COMMAND ----------

# summarize vocabulary
page_categories = list(model.wv.vocab)

# COMMAND ----------

# page categories 
print(page_categories)

# COMMAND ----------

# sample embedding
print(model['reviews'])

# COMMAND ----------

# embedding shape 
model['offers'].shape

# COMMAND ----------

# capturing embedding matrix
X = model[model.wv.vocab]

# COMMAND ----------

# embedding matrix shapee
X.shape

# COMMAND ----------

# run PCA with 2 compopnent to visualize page category embedding
pca = PCA(n_components=2)
result = pca.fit_transform(X)

# COMMAND ----------

# plotting with page-categories
# create a scatter plot of the projection
plt.figure(figsize=(10,10))
plt.scatter(result[:, 0], result[:, 1])

for i,page_category in enumerate(page_categories):
    plt.annotate(page_category,horizontalalignment='right', verticalalignment='top',xy=(result[i, 0], result[i, 1]))

plt.show()

# COMMAND ----------


