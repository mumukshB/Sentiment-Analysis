import numpy as np 
import pandas as pd

train  = pd.read_csv("../input/labeledTrainData.tsv",sep='\t',header=0,quoting=3) #initialize training data (IMDB review dataset)

from bs4 import BeautifulSoup  #remove html tags
from nltk.corpus import stopwords  #words having high frequency like "the"
import re 
def dataClean(string):    # clean data by removing html tags,numbers and stopwords.
    soup = BeautifulSoup(string,"lxml")
    html_gone = soup.get_text()
    numbers_gone = re.sub("[^a-zA-Z]"," ",html_gone)
    words = numbers_gone.lower().split()
    stops = set(stopwords.words("english"))
    stops_gone = [w for w in words if not w in stops]
    return " ".join(stops_gone)


num_reviews = train.review.size
clean_data=[]
for i in range(0,num_reviews):        # for each review in dataset, apply dataclean method.
    clean_data.append(dataClean(train.review.iloc[i]))
    
    
from sklearn.feature_extraction.text import CountVectorizer #countVectorizer initialises an array of frequencies of words.
from sklearn.ensemble import RandomForestClassifier #supervised-learning
vectorizer = CountVectorizer(analyzer="word",tokenizer=None,preprocessor=None,stop_words=None,max_features=5000)
clean_reviews = vectorizer.fit_transform(clean_data)
clean_reviews = clean_reviews.toarray()

forest = RandomForestClassifier(n_estimators=100)    
forest.fit(clean_reviews,train['sentiment'])


test_data = pd.read_csv('../input/testData.tsv',header=0,sep='\t',quoting=3)
clean_test = []
test_num = test_data['review'].size
for i in range(0,test_num):
    clean_test.append(dataClean(test_data.review.iloc[i]))
    
clean = vectorizer.transform(clean_test)
clean = clean.toarray()
predictions = forest.predict(clean) #predict on the trained random-forest
output = pd.DataFrame({'id':test_data['id'],'sentiment':predictions})
output.to_csv('answer.csv',index=False,quoting=3)

#Predictions are 84.6% accurate for binary sentiment.
