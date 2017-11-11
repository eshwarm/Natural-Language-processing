# -*- coding: utf-8 -*-
"""
Created on Fri Sep 22 15:47:58 2017

@author: Eshwar
"""

import nltk , re, string
from nltk.corpus import udhr 
from nltk import bigrams, trigrams
from collections import Counter, defaultdict

from nltk.tokenize import RegexpTokenizer 


english = udhr.raw('English-Latin1') 
french = udhr.raw('French_Francais-Latin1') 
italian = udhr.raw('Italian_Italiano-Latin1') 
spanish = udhr.raw('Spanish_Espanol-Latin1')  

english_train, english_dev = english[0:1000], english[1000:1100] 
french_train, french_dev = french[0:1000], french[1000:1100] 
italian_train, italian_dev = italian[0:1000], italian[1000:1100] 
spanish_train, spanish_dev = spanish[0:1000], spanish[1000:1100]  
english_test = udhr.words('English-Latin1')[0:1000] 
french_test = udhr.words('French_Francais-Latin1')[0:1000]
italian_test = udhr.words('Italian_Italiano-Latin1')[0:1000] 
spanish_test = udhr.words('Spanish_Espanol-Latin1')[0:1000]

def pre_processing(train_data):
    """Text pre-processeing.
    
    Parameters
    ----------
    arg1 : string
        input text
        
    Returns
    -------
    string
        returns processed text

    """
    train_data = re.sub(r"\d",'', train_data) # remove digits
    train_data = train_data.replace('\n', ' ')# removing next line
    train_data = re.findall(r'(\w+)', train_data, re.UNICODE)
    train_data = ' '.join(train_data)
    return train_data
    
def accuracy_calculation(prob_eng_list, prob_fre_list):
    """Calculate Accuracy for two languages.
    
    Parameters
    ----------
    arg1 : list
        list of probobilities for language1.
    arg2 : list
        list of probobilities for language2.        
    Returns
    -------
    float
        returns accuracy

    """    
    eng_greater = 0
    both_equal_to_one = 0
    eng_lesser = 0
    both_equal_to_zero =0    
    for f, b in zip(prob_eng_list, prob_fre_list):
        if(f>b):
            eng_greater+=1
        elif(f<b):
            eng_lesser+=1
        elif((f==1 and b==1)):
            both_equal_to_one+=1
        elif((f==0 and b==0)):
            both_equal_to_zero+=1

    print('    The Accuracy is ', eng_greater/10, '%')
    
def unigram(english_train, french_train, english_test):
    """Implementation of Unigram Model by counting each character and finding
       it's probability by dividing by total count
    
    Parameters
    ----------
    arg1 : string
        Language 1 text data
    arg2 : string
        Language 2 text data

    """ 
    english_model = defaultdict(lambda: 0) #initialize the dictionary with all zero's
    french_model = defaultdict(lambda: 0)  #initialize the dictionary with all zero's
    
    #Preprocessing and Traiing the English text
    english_train = pre_processing(english_train) 
    for letter in english_train.lower():
        if(" " not in letter):
            english_model[(letter)] +=1 

    #Preprocessing and Traiing the French text                           
    french_train = pre_processing(french_train)   
    for letter in french_train.lower():
        if(" " not in letter):
            french_model[(letter)] +=1   
    
    #Calculate probabilities of all letters in English
    total_count = float(sum(english_model.values()))
    for letter in english_model:
        english_model[letter] /= total_count

    #Calculate probabilities of all letters in French
    total_count = float(sum(french_model.values()))            
    for letter in french_model:
        french_model[letter] /= total_count
    
    prob_eng_list = []
    prob_fre_list = []
    #Caluclate probabilites for English test data
    for word in english_test:      
        word = pre_processing(word)
        prob_eng = 1.0
        for letter in word.lower():
            prob_eng = float(prob_eng) * float(english_model[(letter)])
        prob_eng_list.append(prob_eng)  

    #Caluclate probabilites for French test data
    for word in english_test:
        word = pre_processing(word)
        prob_fre = 1.0
        for letter in word.lower():
            prob_fre = float(prob_fre) * float(french_model[(letter)])  
        prob_fre_list.append(prob_fre)

    accuracy_calculation(prob_eng_list, prob_fre_list)     #accuracy calculation                
    
    
def bigram_model(english_train, french_train, english_test):
    """Implementation of Bigram Model by taking two character's and finding
       it's probability.
    
    Parameters
    ----------
    arg1 : string
        Language 1 text data
    arg2 : string
        Language 2 text data
    arg3 : list
        Language 1 test data
    """ 
    
    english_model = defaultdict(lambda: defaultdict(lambda: 0)) #initialize the dictionary with all zero's
    french_model = defaultdict(lambda: defaultdict(lambda: 0))  #initialize the dictionary with all zero's
    
    #Preprocessing and Traiing the English and French text
    english_train = pre_processing(english_train) 
    french_train = pre_processing(french_train)   
    
    for w1, w2 in bigrams(english_train.lower()):
        english_model[(w1)][w2] += 1
    for w1, w2 in bigrams(french_train.lower()): 
        french_model[(w1)][w2] += 1

    #Caluclate probabilites for English train data                      
    for w1 in english_model:
        total_count = float(sum(english_model[w1].values()))
        for w3 in english_model[w1]:
            english_model[w1][w3] /= total_count

    #Caluclate probabilites for French train data        
    for w1 in french_model:
        total_count = float(sum(french_model[w1].values()))
        for w3 in french_model[w1]:
            french_model[w1][w3] /= total_count
            
    prob_eng = 1.0
    prob_fre = 1.0
    
    prob_eng_list = []
    prob_fre_list = []
    
    #Caluclate probabilites for English test data
    for word in english_test:
        word = pre_processing(word)
        word = ' '+ word
        prob_eng = 1.0
        for w1, w2 in bigrams(word.lower()):
            prob_eng = float(prob_eng) * float(english_model[(w1)][w2])
        prob_eng_list.append(prob_eng)  
     
    #Caluclate probabilites for French test data    
    for word in english_test:
        word = pre_processing(word)
        word = ' '+ word + ' '
        #word = ' '+ word 
        prob_fre = 1.0
        for w1, w2 in bigrams(word.lower()):
            prob_fre = float(prob_fre) * float(french_model[(w1)][w2])  
        prob_fre_list.append(prob_fre)

    accuracy_calculation(prob_eng_list, prob_fre_list) #accuracy calculation                

def trigram_model(english_train, french_train, english_test):
    """Implementation of Trigram Model by taking three character's and finding
       it's probability.
    
    Parameters
    ----------
    arg1 : string
        Language 1 text data
    arg2 : string
        Language 2 text data
    arg3 : list
        Language 1 test data
    """    
    english_model = defaultdict(lambda: defaultdict(lambda: 0)) #initialize the dictionary with all zero's
    french_model = defaultdict(lambda: defaultdict(lambda: 0))  #initialize the dictionary with all zero's
    
    #Preprocessing and Traiing the English and French text
    english_train = pre_processing(english_train)
    french_train = pre_processing(french_train)

    for w1, w2, w3 in trigrams(english_train.lower()):
        english_model[(w1, w2)][w3] += 1
    for w1, w2, w3 in trigrams(french_train.lower()): 
        french_model[(w1, w2)][w3] += 1
    
    #Caluclate probabilites for English and French train data                     
    for w1_w2 in english_model:
        total_count = float(sum(english_model[w1_w2].values()))
        for w3 in english_model[w1_w2]:
            english_model[w1_w2][w3] /= total_count
            
    for w1_w2 in french_model:
        total_count = float(sum(french_model[w1_w2].values()))
        for w3 in french_model[w1_w2]:
            french_model[w1_w2][w3] /= total_count
            
    prob_eng = 1.0
    prob_fre = 1.0
    
    prob_eng_list = []
    prob_fre_list = []
    
    #Caluclate probabilites for English test data
    for word in english_test:
        #word = ' '+ word + ' '
        word = pre_processing(word)
        #word = ' '+ word
        prob_eng = 1.0
        for w1, w2, w3 in trigrams(word.lower()):
            prob_eng = float(prob_eng) * float(english_model[(w1, w2)][w3])
        prob_eng_list.append(prob_eng)   
    #Caluclate probabilites for French test data
    for word in english_test:
        word = pre_processing(word)
        #word = ' '+ word + ' '
        word = ' '+ word 
        prob_fre = 1.0
        for w1, w2, w3 in trigrams(word.lower()):
            prob_fre = float(prob_fre) * float(french_model[(w1, w2)][w3])  
        prob_fre_list.append(prob_fre)

    accuracy_calculation(prob_eng_list, prob_fre_list) #accuracy calculation



if __name__ == "__main__":
    print("Problem 1** ...")
    print("\n    English Vs French    ")
    print("    -----------------    \n")
    print("Trigram:")
    trigram_model(english_train, french_train, english_test)
    print("Bigram:")
    bigram_model(english_train, french_train, english_test)
    print("Unigram:")
    unigram(english_train, french_train, english_test)
    
    print("\nProblem 2** ...")
    print("\n    Spanish Vs Italian    ")
    print("    -----------------    \n")
    print("Trigram:")
    trigram_model(spanish_train, italian_train, spanish_test)
    print("Bigram:")
    bigram_model(spanish_train, italian_train, spanish_test)
    print("Unigram:")
    unigram(spanish_train, italian_train, spanish_test)    
    