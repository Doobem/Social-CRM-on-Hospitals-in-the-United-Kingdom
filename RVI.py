# -*- coding: utf-8 -*-
"""
Created on Wed Jul 20 20:23:12 2022

@author: chikk
"""



#import nltk
#nltk.download('vader_lexicon')
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer

#import tweets
Reviews = pd.read_excel('Ratings and reviews - The Royal Victoria Infirmary - NHS.xlsx')
print (Reviews)

#call the function
sia = SentimentIntensityAnalyzer()

#apply sia and transform them into the dataframe
Reviews['neg'] = Reviews['Comment'].apply(lambda x:sia.polarity_scores(x)['neg'])
Reviews['neu'] = Reviews['Comment'].apply(lambda x:sia.polarity_scores(x)['neu'])
Reviews['pos'] = Reviews['Comment'].apply(lambda x:sia.polarity_scores(x)['pos'])
Reviews['compound'] = Reviews['Comment'].apply(lambda x:sia.polarity_scores(x)['compound'])

pos_Reviews = [j for i, j in enumerate(Reviews['Comment']) if Reviews['compound'][i] > 0.2]
neu_Reviews = [j for i, j in enumerate(Reviews['Comment'])if 0.2>=Reviews['compound'][i]>=-0.2]
neg_Reviews = [j for i, j in enumerate(Reviews['Comment'])if Reviews['compound'][i]< -0.2]

print()


print ("percentage of positive Reviews:{}%".format(len(pos_Reviews)*100/len(Reviews['Comment'])))
print ("percentage of neural Reviews:{}%".format(len(neu_Reviews)*100/len(Reviews['Comment'])))
print ("percentage of negative Reviews:{}%".format(len(neg_Reviews)*100/len(Reviews['Comment'])))

#sample reuslt
print(Reviews.head())

from textblob import TextBlob
feedback = str(Reviews)
ob = TextBlob(feedback)
print(ob.sentiment.polarity)         
print(ob.sentiment.subjectivity)    
print(ob.sentiment)


