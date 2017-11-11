import gensim, pprint
import logging
import pickle 
import operator, numpy
import collections
from gensim import utils, matutils
from numpy import float32, array, dot

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
model = gensim.models.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)

def get_questions():
	"""
	read anaologies from the question-words text file.
	make a dictionary by type of anaologies
	"""
    with open('questions-words.txt', 'r+') as f:
        lines  = f.readlines()
    questions_dict = collections.defaultdict(list)
    key = ''
    for line in lines:
        if(':' in line):
            key = line.split(':')[1]
        else:
            questions_dict[key.strip()].extend([line.strip().split()])
    return questions_dict


def get_numpy_array(weight, word):
    """
    Returns the word's representations in vector space, as a 1D numpy array.
    """
    return weight * model.word_vec(word, use_norm=True)

def get_unit_vector(mean):
    """
    Scale a vector to unit length. The only exception is the zero vector, which
    is returned back unchanged.
    """
    return matutils.unitvec(array(mean).mean(axis=0)).astype(float32)

def get_top_n_words(vectors, n):
    """
     If reverse is True, return the greatest elements instead, in descending order.
     """
    return matutils.argsort(vectors, 13, reverse=True)
        
def find_similar(positive, negative):
	"""
	By cosine similarity find the dot product of each anaology by all pretrained
	vectors in the word2vec model.
    
	Parameters
    ----------
    arg1 : list
        positive words	
    arg2 : list
        negative words		
	Returns
    -------
    List
        returns top ten words of similarity words
	"""
    model.init_sims()
    pos = [(word, 1.0) for word in positive]
    neg = [(word, -1.0) for word in negative]

    # compute the weighted average of all words
    mean = []
    all_words = set()
    for word, weight in pos + neg:
        
        result = get_numpy_array(weight, word)
        mean.append(result)
        if word in model.vocab:
            all_words.add(model.vocab[word].index)
    
    mean = get_unit_vector(mean)
    
    dot_product_all_vectors = dot(model.syn0norm, mean)
    top_indexes = get_top_n_words(dot_product_all_vectors, 10)
    result = [model.index2word[index] for index in top_indexes if index not in all_words]
    return result[:10]    
	
def accuracy(questions):
	"""
	Find accuracy for the test anology data.
	Parameters
    ----------
    arg1 : dictionary
        dictionary of anology question with each type.
	"""
	for key, list_ in questions.items():
		total = 0
		predicted_count = 0
		for question in list_:
			max = 0.0
			predicted = ''
			dic_ = {}       
			a, b, c, d = question
			positive = [b,c]    
			negative = [a]
			expected = d
			ignore = {a, b, c}
			ok_vocab = [(w, model.vocab[w]) for w in model.index2word[:30000]]
			ok_vocab = {w.upper(): v for w, v in reversed(ok_vocab)} if False else dict(ok_vocab)        
			if a not in ok_vocab or b not in ok_vocab or c not in ok_vocab or d not in ok_vocab:
				continue
			re = find_similar(positive, negative)
			total+=1	
			if(re[0]== expected):
				predicted_count+=1				
		print('Total:',total, 'predicted:',predicted_count)
		print('Accuracy for ', key, (predicted_count/total)*100)
		
if __name__ == "__main__":
	questions = get_questions()
	accuracy(questions)


            
         




