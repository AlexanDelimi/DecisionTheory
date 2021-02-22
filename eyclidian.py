import pandas
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
import time
import tensorflow
from keras.callbacks import EarlyStopping
from math import*                                       #imports
 
def euclidean_distance(x,y):
 
    return sqrt(sum(pow(a-b,2) for a, b in zip(x, y)))  #Definition of eyclidian distance between two vectors of same size

start_time = time.time()    # Start the time.

ps = PorterStemmer()    # Initialize the stemmer.
tf_idf = TfidfVectorizer()  # Initialize tf-idf.
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)   # Initialize early stopping.
stop_words = set(stopwords.words('english'))    # Set language for stop words.

filen = pandas.read_csv("./SocialMedia_Negative.csv")
filep = pandas.read_csv("./SocialMedia_Positive.csv")   #Import the two files


def tfidf(file):                #Begin of making the combined tf-idf vector of the median values of all documents inside the file
    text = file.Text
    labels = file.Sentiment        #Divide between text document and text labels

    for i,label in enumerate(labels):
        if label == 'negative':
            labels[i] = 0.0
        else:
            labels[i] = 1.0             #Binary numerical represantation of the text labels




    vector_text = text.to_numpy() #Convert the text list to numpy
    
    vectors_of_words = []
    for strings in range(len(vector_text)):     
        
        vector_text[strings] = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', vector_text[strings], flags=re.MULTILINE)
        vector_text[strings] = re.sub("[^a-zA-Z0-9 ]", "",vector_text[strings])
        vector_text[strings] = vector_text[strings].lower()
        for word in word_tokenize(vector_text[strings]):    
            new_word = ps.stem(word)
            vector_text[strings] = vector_text[strings].replace(word, new_word)
            if new_word in stop_words:  
                vector_text[strings] = vector_text[strings].replace(word, "")
        vector_text[strings] = re.sub(' +',' ',vector_text[strings])                    #Filter the text list, removing stopwords, caps etc.
        

    x = tf_idf.fit(vector_text)     #Getting the tf-idf matrix
    vocab = x.vocabulary_           #Getting the vocab dictionary to correctly map the tf-idf
   

    x = tf_idf.transform(vector_text)   # Executes the tf-idf transformation.
    x = x.toarray()                     #Transform the matrix into an array of arrays for easier reiterative manipulation
    
    x_array = np.zeros(len(x[1]))       #Define an array to hold the sum of each tf-idf of each word in all documents
    x_length = np.zeros(len(x[1]))      #Define a support array to count the words for correct median calculation
    for i in range(len(x)):             #For every element of x i.e. every tf-idfed document
        for j in range(len(x[i])):      #For every point in the tf-idf space
            x_array[i] = x_array[i] + x[i][j]   #Add to the sum array the value of the point
            x_length[i] = x_length[i] + 1       #And ++ the count vector 
        vocab[i] = x_array[i]/x_length[i]       #Replace in the vocablurary the count of the words with median value

    return vocab                    #And return the dictionary 

positive = tfidf(filep)
negative = tfidf(filen) #Do the above process for both files

def compare_vectors(vector_1,vector_2):     #Function to calculate the euclidian distance of the projection of the first vector into the second
    temp_vector_2 = vector_1.copy()         #Make a temporary first vector which will be the projection of the second into the first's space
    for key in temp_vector_2:               #For each point of the space of the first vector
        temp_vector_2[key] = 0.0000000000000000000  #Replace the value with zero so as to initialize it
    for key in vector_1:                    #For each point of the space of the first vector
        if str(key) in vector_2.keys():     #If there is something to project
            temp_vector_2[key] = vector_2[key]      #Do it
    vec_1 = list(vector_1.values())         
    vec_2 = list(temp_vector_2.values())        #Get the values of the dictionaries in the form of a list
    print(euclidean_distance(vec_1,vec_2))      #And print the eucldian distance of the two

compare_vectors(positive,negative)   
compare_vectors(negative,positive)      #DO the process for both ways
