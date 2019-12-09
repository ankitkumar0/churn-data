# -*- coding: utf-8 -*-
"""
Created on Wed Sep 25 13:45:12 2019

@author: Deepak Gupta
"""

import nltk
import os
import nltk.corpus
os.chdir("E:\\PYTHON NOTES\\file")
#Load Text data
nltk.corpus.gutenberg.fileids()
hmlet = nltk.corpus.gutenberg.words("shakespeare-hamlet.txt")
for word in hmlet[:500]:
    print(word,sep=" ",end=" ")
    
from nltk.tokenize import word_tokenize    

al= """The Tragedie of Hamlet by William Shakespeare 1599 ] Actus Primus . Scoena Prima . Enter Barnardo and Francisco two Centinels . Barnardo . Who ' s there ? Fran . Nay answer me : Stand & vnfold your selfe Bar . Long liue the King Fran . Barnardo ? Bar . He Fran . You come most carefully vpon your houre Bar . ' Tis now strook twelue , get thee to bed Francisco Fran . For this releefe much thankes : ' Tis bitter cold , And I am sicke at heart Barn . Haue you had quiet Guard ? Fran . Not a Mouse stirring Barn . Well , goodnight . If you do meet Horatio and Marcellus , the Riuals of my Watch , bid them make hast . Enter Horatio and Marcellus . Fran . I thinke I heare them . Stand : who ' s there ? Hor . Friends to this ground Mar . And Leige - men to the Dane Fran . Giue you good night Mar . O farwel honest Soldier , who hath relieu ' d you ? Fra . Barnardo ha ' s my place : giue you goodnight . Exit Fran . Mar . Holla Barnardo Bar . Say , what is Horatio there ? Hor . A peece of him Bar . Welcome Horatio , welcome good Marcellus Mar . What , ha ' s this thing appear ' d againe to night Bar . I haue seene nothing Mar . Horatio saies , ' tis but our Fantasie , And will not let beleefe take hold of him Touching this dreaded sight , twice seene of vs , Therefore I haue intreated him along With vs , to watch the minutes of this Night , That if againe this Apparition come , He may approue our eyes , and speake to it Hor . Tush , tush , ' twill not appeare Bar . Sit downe a - while , And let vs once againe assaile your eares , That are so fortified against our Story , What we two Nights haue seene Hor . Well , sit we downe , And let vs heare Barnardo speake of this Barn . Last night of all , When yond same Starre that ' s Westward from the Pole Had made his course t ' illume that part of Heauen Where now it burnes , Marcellus and my selfe , The Bell then beating one Mar . Peace , breake thee of : Enter the Ghost . Looke where it comes againe Barn . In the same figure , like the King that ' s dead Mar . Thou art a Scholler ; speake to it Horatio Barn . Lookes it not like the King ? Marke it Horatio Hora . Most like : It harrowes me with fear & wonder Barn . It would be spoke too Mar . Question it Horatio Hor . What art """
type(al)
al_token=word_tokenize(al)
len(al_token)
from nltk.probability import FreqDist
freqdist=FreqDist()

for word in al_token:
    freqdist[word.lower()]+=1
    
freqdist    

fdist_top10=freqdist.most_common(10)
from nltk.tokenize import blankline_tokenize
al_blank=blankline_tokenize(al)
len(al_blank)
from nltk.util import bigrams,trigrams,ngrams
strings="Marke it Horatio Hora . Most like : It harrowes me with fear & wonder Barn . It would be spoke too Mar . Question it Horatio Hor . What art "
qutoes_token=nltk.word_tokenize(strings)
qutoes_bigrams=list(nltk.bigrams(qutoes_token))
qutoes_trigrams=list(nltk.trigrams(qutoes_token))
qutoes_ngrams=list(nltk.ngrams(qutoes_token,5))
#stemmer
from nltk.stem import PorterStemmer
ps= PorterStemmer()
ps.stem("speaking")
word_to_strem=["give","given","gave"]
for word in word_to_strem:
    print(word+ ":"+ps.stem(word))

from nltk.stem import LancasterStemmer
lan=LancasterStemmer()
for word in word_to_strem:
    print(word+ ":" +lan.stem(word))
    
from nltk.stem import wordnet    
from nltk.stem import WordNetLemmatizer    
word_len=WordNetLemmatizer()    
for word in word_to_strem:
   print(word+ ":" +word_len.lemmatize(word))   

from nltk.corpus import stopwords 
stop = set(stopwords.words("english"))    
len(stop)    
import re
punctuation=re.compile(r'[-.?!,:;()|0-9]')
post_punctuation=[]
for word in al_token:
    words=punctuation.sub("",word)
    if len(words)>0:
        post_punctuation.append(words)
        

len(post_punctuation)
 #pos tag & discription
sent="you are one and only my best friend in this city or in the world"
sent_token=word_tokenize(sent)
for token in sent_token:
    print(nltk.pos_tag([token]))

#enetity regonation
from nltk import ne_chunk
ne_send ="The US Precident Stay in the White house"
sent_token=word_tokenize(ne_send)
ne_tag=nltk.pos_tag(sent_token)

ne_nrr=ne_chunk(ne_tag)

new="The cat ate the little mouse who was after fresh cheese"
new_token=nltk.pos_tag(word_tokenize(new))
grammer_np=r"NP: {<DT>?<JJ>*<NN>}"
chunk_praser=nltk.RegexpParser(grammer_np)    
chunk_result=chunk_praser.parse(new_token)
import pandas as pd    
import numpy as np    
import sklearn

from sklearn.feature_extraction.text import CountVectorizer
    
from nltk.corpus import movie_reviews    
movie_reviews.categories()    
pos_rev=movie_reviews.fileids("pos")
neg_rev=movie_reviews.fileids("neg")
rev=nltk.corpus.movie_reviews.words('pos/cv565_29572.txt')    
rev_list=[]  
  
for rev in neg_rev:
    rev_text_neg=rev=nltk.corpus.movie_reviews.words(rev)
    review_one_string=" ".join(rev_text_neg)
    review_one_string=review_one_string.replace(",",",")
    review_one_string=review_one_string.replace(".",".")
    review_one_string=review_one_string.replace("\' ","'")
    review_one_string=review_one_string.replace(" \'","'")
    rev_list.append(review_one_string)
    
len(rev_list)    
for rev in pos_rev:
    rev_text_neg=rev=nltk.corpus.movie_reviews.words(rev)
    review_one_string=" ".join(rev_text_neg)
    review_one_string=review_one_string.replace(",",",")
    review_one_string=review_one_string.replace(".",".")
    review_one_string=review_one_string.replace("\' ","'")
    review_one_string=review_one_string.replace(" \'","'")
    rev_list.append(review_one_string)
        
neg_target=np.zeros((1000,),dtype=np.int) 
pos_target=np.ones((1000,),dtype=np.int) 

target_list=[]   
     
for neg_tar in neg_target:
    target_list.append(neg_tar)
for pos_tar in pos_target:
    target_list.append(pos_tar)
        
len(target_list)    
    
y=pd.Series(target_list) 
   
from sklearn.feature_extraction.text import CountVectorizer
count_vec=CountVectorizer(lowercase=True,stop_words="english",min_df=2)
x_count_vec=count_vec.fit_transform(rev_list)
x_count_vec.shape
x_name=count_vec.get_feature_names()
x_name
x_count_vec=pd.DataFrame(x_count_vec.toarray(),columns=x_name)
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
x_train_cv,x_test_cv,y_train_cv,y_test_cv=train_test_split(x_count_vec,y,test_size=0.25,random_state=5)
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
y_pred_nb=gnb.fit(x_train_cv,y_train_cv).predict(x_test_cv)
from sklearn.naive_bayes import MultinomialNB
clf_vc=MultinomialNB()
clf_vc.fit(x_train_cv,y_train_cv)
y_pred_cv=clf_vc.predict(x_test_cv)
type(y_pred_cv)
print(metrics.accuracy_score(y_test_cv,y_pred_cv)) 
score_clf_cv=confusion_matrix(y_test_cv,y_pred_cv)





