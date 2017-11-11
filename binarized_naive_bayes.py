import collections, re, math

all_words = []  #list of all words
Vocabulary = [] #list of all unique vocabulary
global n_doc
n_doc_class_words_dict = collections.defaultdict(lambda:0)  #total no of documents per class
class_words_dict = collections.defaultdict(list)  #dictionary to hold all words per class

class Classifier():
    """
    Classifier class which calculates the 
    loglikelihood of all words per class category
    """
    def __init__(self, class_, n_doc):
        self.class_ = class_      #class name
        self.total_doc_count = n_doc  #total documents count
        self.class_doc_count = n_doc_class_words_dict[class_]   #total documents per class
        self.all_words = class_words_dict[self.class_]
        self.V = len(Vocabulary)  #length of total unique vocabulary
        self.logprior = math.log((self.class_doc_count/self.total_doc_count),10) #logprior calculation
        self.count = collections.defaultdict(lambda:0)
        self.loglikelihood = collections.defaultdict(lambda:0)
        self.calculate_loglikelihood()
        
    def calculate_loglikelihood(self):
        """
        Calculate log likliehood 
        Here aplha value is 0.3
        """
        for word in list(set(self.all_words)):
            count  = self.get_word_count(word)
            self.count[word] = count
            self.loglikelihood[word] = (self.count[word]+0.3)/((len(self.all_words)+(0.3*(self.V))))
            
    def get_word_count(self, word):
        """
        get the count of each word
        """
        count = 0
        val = [1 for each_word in self.all_words if each_word == word]   
        return sum(val)
        
def read_data(type_of_data):
    """
    read the contents of file
    """
    with open('../data/{0}'.format(type_of_data), 'r+') as t:
        lines  = t.readlines()
    return lines

def data_processing(line):
    """
    Pre-processing data: removing special characters and digits
    """
    sentence_list = line.split(' ')
    class_name = sentence_list[0]
    del sentence_list[0]   
    
    line = ' '.join(sentence_list) 
    line = re.sub(r"\d",'', line) # remove digits
    line = line.replace('\n', ' ')# removing next line
    line = re.findall(r'(\w+)', line, re.UNICODE)
    return class_name, line   

def get_class_words_dict():
    """
    reading all lines and storing words per class wise in a dictionary
    """
    n_doc = 0
    lines  = read_data('train')
    for line in lines:
        n_doc += 1   #calculating document count
        class_name, sentence_list = data_processing(line)
        n_doc_class_words_dict[class_name]+=1    #calculating document count per class
        all_words.extend(list(set(sentence_list))) #storing all words
        class_words_dict[class_name].extend(list(set(sentence_list))) #storing words per class in a dictionary  
    Vocabulary.extend(list(set(all_words)))
    return n_doc
    
    
def test_naive_bayes(object_dict):
    """
    Take test data and finding accuracy
    """
    lines  = read_data('test')
    acc_count =0
    line_count = 0
    for line in lines:
        line_count +=1
        sum_dict = collections.defaultdict(lambda:0)
        sum = 0
        class_name, sentence_list = data_processing(line)
        for class_, ob in object_dict.items():  #For each object per instance of class
            sum = ob.logprior  #logproior of  a class
            for word in sentence_list:
                if word in ob.all_words:
                    sum += math.log(ob.loglikelihood[word])  #sum of loglikliehood
                else:
                    sum+= math.log(1/(len(ob.all_words)+(ob.V*0.3)))
            sum_dict[class_] = sum
        result = max(sum_dict, key=sum_dict.get)  #maximum of all values
        if (class_name == result):
            acc_count+=1       #value to increment the count of match with test
    print("The Acurracy of Binary Naïve Bayes classifier is",(acc_count/line_count)*100)   #total accuracy 
    

if __name__ == "__main__":
    n_doc = get_class_words_dict() 
    object_dict = collections.defaultdict(object)  #initialising the dictionary for objects of all classes
     
    for class_, list_ in class_words_dict.items():
        ob = Classifier(class_, n_doc ) 
        object_dict[class_] = ob
    test_naive_bayes(object_dict)    #test the data
