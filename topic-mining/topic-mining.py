# Databricks notebook source
# check if spark context is defined
print(sc.version)

# importing some libraries
import pandas as pd
import pyspark
from pyspark.sql import SQLContext

sqlContext = SQLContext(sc)
# stuff we'll need for text processing
# from nltk.corpus import stopwords
import re as re
from pyspark.ml.feature import CountVectorizer, IDF
# stuff we'll need for building the model

from pyspark.ml.linalg import Vector, Vectors
from pyspark.ml.clustering import LDA, LDAModel

# reading the data
data = sqlContext.read.format('text').options(header='false', inferschema='true').load(
    "/FileStore/tables/dataset_1605467")
data.printSchema()

from pyspark.ml.feature import StopWordsRemover

reviews = data.filter(data['value'].isNotNull())

# Tokenization
from pyspark.ml.feature import Tokenizer

tokenization = Tokenizer(inputCol='value', outputCol='tokens')
tokenized_df = tokenization.transform(reviews)
tokenized_df.printSchema()

# stopwords removal

from pyspark.ml.feature import StopWordsRemover

spanish_swords = StopWordsRemover.loadDefaultStopWords('spanish')
stopword_removal = StopWordsRemover(inputCol='tokens', outputCol='refined_tokens').setStopWords(spanish_swords)
refined_df = stopword_removal.transform(tokenized_df)

# Count Vectorizer
from pyspark.ml.feature import CountVectorizer

count_vec = CountVectorizer(inputCol='refined_tokens', outputCol='tf_features', vocabSize=5000, minDF=10.0)
result_cv = count_vec.fit(refined_df).transform(refined_df)

# result_cv.select(['Clothing ID','refined_tokens','raw_features']).show(4,False)
vocabArray = count_vec.fit(refined_df).vocabulary

# Tf-idf
# from pyspark.ml.feature import HashingTF,IDF

# hashing_vec=HashingTF(inputCol='refined_tokens',outputCol='tf_features')
# hashing_df=hashing_vec.transform(refined_df)
# hashing_df.select(['refined_tokens','tf_features']).show(4,False)

tf_idf_vec = IDF(inputCol='tf_features', outputCol='features')

tf_idf_df = tf_idf_vec.fit(result_cv).transform(result_cv)
# tf_idf_df.select(['user_id','tf_idf_features']).show(4,False)

# tf_idf_df.cache()
tf_idf_df.persist(storageLevel=pyspark.StorageLevel.MEMORY_AND_DISK)

from pyspark.ml.clustering import LDA, LDAModel

num_topics = 10
max_iterations = 10
lda = LDA(k=num_topics, maxIter=max_iterations)
# lda_model = lda.fit(tf_idf_df[['index','features']].rdd.map(list))


model = lda.fit(tf_idf_df)

ll = model.logLikelihood(tf_idf_df)
lp = model.logPerplexity(tf_idf_df)
print("The lower bound on the log likelihood of the entire corpus: " + str(ll))
print("The upper bound on perplexity: " + str(lp))

# Describe topics.
topics = model.describeTopics(3)
print("The topics described by their top-weighted terms:")
topics.show(truncate=False)

# Shows the result
# transformed = model.transform(tf_idf_df)
# transformed.show(truncate=False)


wordNumbers = 5
topicIndices = model.describeTopics(maxTermsPerTopic=wordNumbers)


def topic_render(topic):
    terms = topic[0]
    result = []
    for i in range(wordNumbers):
        term = vocabArray[terms[i]]
        result.append(term)
    return result


topicos = topicIndices.select('topic').collect()
# print(topicos)
tindices = topicIndices.select('termIndices').collect()
# print(tindices)

print(len(topicos))
for topic in range(len(topicos)):
    print("Topic " + str(topic) + ":")
    print(topic_render(tindices[topic]))
    '''for term in tindices[topic]:
        print(term)'''




