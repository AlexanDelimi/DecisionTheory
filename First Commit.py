import pandas #Η βιβλιοθήκη που μας επιτρέπει να χειριζόμαστε τα δεδομένα εύκολα
import re #Η βιβλιοθήκη για τα regular expressions
import numpy #Η βιβλιοθήκη για την μορφοποιήση της εισόδου στο νευρωνικό
from sklearn.feature_extraction.text import TfidfVectorizer #Η βιβλιοθήκη για την υλοποιήση του tf-idf
from sklearn.metrics import f1_score, precision_score, recall_score # Οι μετρικές μας
from sklearn.model_selection import train_test_split #Για τον διαχωρισμό του συνόλου εκπαίδευσης και επαλήθευσης
from nltk.corpus import stopwords #Οι συνήθεις λέξεις που θα αφαιρεθούν από το κείμενο
from nltk.stem import PorterStemmer # Η διαδικασία του Stemming
from nltk.tokenize import word_tokenize #H διαδικασία της τμηματοποίησης των κειμένων-προτάσεων σε λέξεις
import time # Η βιβλιοθήκη για την μέτρηση του χρόνου
import tensorflow # η βιβλιοθήκη που έχει υλοποιημένο το νευρωνικό μας
from keras.callbacks import EarlyStopping #Να σταματάει αν τα αποτελέσματα ειναι ικανοποιητικά

start_time = time.time()    # Εκκίνηση χρονομέτρησης

ps = PorterStemmer()    # Αρχικοποίηση του stemmer
tf_idf = TfidfVectorizer()  # Αρχικοποίηση του tf-idf.
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1)   # Αρχικοποίηση της πρόωρης παύσης
stop_words = set(stopwords.words('english'))    # Επιλογή γλώσσας για τα stopwords

filen = pandas.read_csv("./SocialMedia_Negative.csv") #Φόρτωση του αρχείου για τα αρνητικά σχόλια
filep = pandas.read_csv("./SocialMedia_Positive.csv") #Φόρτωση του αρχείου για τα θετικά σχόλια
file = pandas.concat([filen, filep], axis=0, ignore_index=True) #Συνένωση των δύο αρχείων
text = file.Text #Βαζουμε στην λίστα text το κείμενο των αρχείων
labels = file.Sentiment#και στην λίστα labels το sentiment των αρχείων

for i,label in enumerate(labels): #Διατρέχουμε την λίστα
    if label == 'negative': #Αντικαθιστουμε τα negative
        labels[i] = 0.0 #με 0.0
    else: #και τα θετικα
        labels[i] = 1.0 # με 1.0

vector_text = text.to_numpy() #Μετατρέπουμε το κείμενο σε μορφή numpy
vectors_of_words = [] #Αρχικοποιούμε το διάνυσμα που θα έχει όλες τις λέξεις.
for strings in range(len(vector_text)):     # Για καθε προταση του vector_text
    vector_text[strings] = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', '', vector_text[strings], flags=re.MULTILINE) #Διαγραφουμε τους ειδικολυς χαρακτήρες
    vector_text[strings] = re.sub("[^a-zA-Z0-9 ]", "",vector_text[strings])
    vector_text[strings] = vector_text[strings].lower()# Κάνουμε όλα τα γράμματα πεζά
    for word in word_tokenize(vector_text[strings]):    #Για καθε λεξη
        new_word = ps.stem(word) #Κάνε stem
        vector_text[strings] = vector_text[strings].replace(word, new_word)#Αντικατάστησε τις λέξεις με το αποτέλεσμα του stem τους
        if new_word in stop_words:  # Για κάθε λέξη στα stopwords
            vector_text[strings] = vector_text[strings].replace(word, "") #Αφαίρεσε το συγκεκριμένο stopword
    vector_text[strings] = re.sub(' +',' ',vector_text[strings])

x = tf_idf.fit(vector_text)#Προετοίμασε το tf-idf


x = tf_idf.transform(vector_text)   # Εκτέλεσε το tf-idf

df = pandas.DataFrame(x.toarray(), columns=tf_idf.get_feature_names()) #Επέστρεψε τα ονόματα των χαρακτηριστικών
print("\n\n\n")
df.insert(len(df.columns), "labelz", labels, True)   # Προσθεσε τις ετικέτες στο dataframe df

X_train, X_test, y_train, y_test = train_test_split(df.loc[:, df.columns != 'labelz'], df.labelz, test_size=0.25)   #Χώρισε τα σύνολα εκπαίδευσης και σύνολα επαλήθευσης


num_y_train = y_train.to_numpy().astype('float32')#
num_y_test = y_test.to_numpy().astype('float32')#Μετατρέπουμε τα δεδομένα μας σε float
num_x_test = X_test.to_numpy().astype('float32')#
num_x_train = X_train.to_numpy().astype('float32')#

model = tensorflow.keras.models.Sequential()    # Αρχικοποιήση Μοντελου
model.add(tensorflow.keras.layers.Flatten())    # Μετατροπή του σε μονοδιάστατο πίνακα
model.add(tensorflow.keras.layers.Dense(128, activation=tensorflow.nn.relu))    # Προσθήκη επιπέδου
model.add(tensorflow.keras.layers.Dense(128, activation=tensorflow.nn.relu))    #Προσθήκη επιπέδου
model.add(tensorflow.keras.layers.Dense(2, activation=tensorflow.nn.softmax))   # Επίπεδο εξόδου, μέγεθος όσο οι πιθανές απαντήσεις
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])   # Μεταγλώτισση μοντέλου
model.fit(num_x_train, num_y_train, epochs=10, batch_size=16, verbose=1, validation_data=(num_x_test, num_y_test), callbacks=[es])  # Ταίριασμα μοντέλου

val_loss, val_acc = model.evaluate(num_x_test, num_y_test )  # Αξιολόγηση μοντέλου
print("\n\nLoss: \t\t", val_loss, "\nAccuracy: \t", val_acc, "\n\n")#Εκτύπωση μετρικών


predictions = model.predict(num_x_test)  # Προβλέψεις μοντέλου.


predictions_list = list()#Αρχικοποιήση λίστας προβλέψεων
for k in range(len(predictions)):   # Για κάθε πρόβλεψη
    predictions_list.append(numpy.argmax(predictions[k]))   # Προσθέτουμε την πρόβλεψη στην λίστα προβλέψεων


print("F1 Score: \t\t\t", round(f1_score(num_y_test , predictions_list, average='micro'), 4))
print("Precision Score: \t", round(precision_score(num_y_test , predictions_list, average='micro'), 4)) # υπολογισμός και εκτύπωση μετρικών
print("Recall Score: \t\t", round(recall_score(num_y_test , predictions_list, average='micro'), 4), "\n\n")

print("--- %s seconds ---" % (time.time() - start_time))    #Τέλος χρονομέτρησης και τέλος υλοποιήσης
