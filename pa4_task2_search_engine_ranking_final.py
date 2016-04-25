#####begin put all your imports
# encoding=utf8
# -*- coding: utf-8 -*-
#u"åäö".encode('utf-8') 
import nltk

import pymongo
import os
import string
import re
from bs4 import BeautifulSoup
from collections import defaultdict, Counter
from pymongo import MongoClient
from nltk.corpus import stopwords
from nltk import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import RegexpTokenizer
from nltk.tokenize import word_tokenize
import operator
import math
import numpy as np
from sklearn.metrics.pairwise import linear_kernel
import sklearn as sk
import pickle
import csv
#####end put all your imports 


mongodb_client = MongoClient('localhost',27017)
mongodb_db = mongodb_client['uta-edu-corpus']
document_frequency = defaultdict(int) 
total_number_of_docs = 141

def setup_mongodb():
    #####################Task t2a: your code #######################
    #Connect to mongodb 
	#connection=pymongo.Connection()
 #   print(u'\2019')
    mongodb_client = MongoClient('localhost',27017)
    mongodb_db = mongodb_client['uta-edu-corpus'] 
    print mongodb_client.database_names()
    #####################Task t2a: your code #######################
   
#This function processes the entire document corpus
def process_document_corpus(file_name):
    #####################Task t2b: your code #######################
    #### The input is a file where in each line you have two information
    #   filename and url separated by |
    # Process the file line by line
    #   and for each line call the function process_document with file name and url and index
    #   first file should have an index of 0, second has 1 and so on
    #Remember to set total_number_of_docs to number of documents
    #####################Task t2b: your code #######################
	#file_name="fileNamesToUUID.txt"
	
	#full_path = os.path.realpath(file_name)
	#file_path = '%s/fileNamesToUUID.txt' % os.path.dirname(full_path)
	
	with open(file_name) as myfile:
		i=0
		for line in myfile:
			print "\n \n *****line********"
			print line
			words = [item.strip() for item in line.split('|')]
			process_document(words[0],words[1],i)
			i=i+1
		pass 


#This function processes a single web page and inserts it into mongodb
def process_document(file_name, url, index):
    #Do not change 
	
    f = open(file_name)
    #print 'hello'
    file_contents = f.read()
    f.close()
	
    soup = BeautifulSoup(file_contents)
    #soup = BeautifulSoup (file_contents.encode('utf-8', 'ignore'))


    #####################Task t2c: your code #######################
    #Using the functions that you will write (below), convert the document
    #   into the following structure:
    # title_processed: a string that contains the title of the string after processing
    # hx_processed: an array of strings where each element is a processed string
    #   for eg, if the document has two h1 tags, then the array h1_processed will have two elements
    #   one for each h1 tag and contains its content after processing
    # a_processed: same for a tags
    # body_processed: a string that contains body of the document after processing
    print "Title********"
    title=soup.title.text
    print title.encode('ascii','ignore')
    print "Title end*********"

    print "Title Processed*********"
    title_processed = process_text(title)
    print "Title Processed end********"
    
    print "h1 processed********"
    if(soup.find_all("h1")==[]):
	h1_processed=[]
    else:
	#h1_processed=process_text(soup.h1.text)
        h1_processed = process_array(soup.find_all("h1"))
    print "h1 processed end**********"

    print "h2 processed********"
    if(soup.find_all("h2")==[]):
	    h2_processed=[]
    else:
	#h2_processed=process_text(soup.h2.text)
        h2_processed = process_array(soup.find_all("h2"))
    print "h2 processed end**********"

    print "******h3 processed ********"
    if(soup.find_all("h3")==[]):
	    h3_processed=[]
    else:
	#h3_processed=process_text(soup.h3.text)
        h3_processed = process_array(soup.find_all("h3"))
    print "******h3 processed end ********"

    print "******h4 processed ********"
    if(soup.find_all("h4")==[]):
	    h4_processed=[]
    else:
	#h4_processed=process_text(soup.h4.text)
        h4_processed = process_array(soup.find_all("h4"))
    print "******h4 processed end ********"

    print "******h5 processed********"
    if(soup.find_all("h5")==[]):
	    h5_processed=[]
    else:
	    #h5_processed=process_text(soup.h5.text)
        h5_processed = process_array(soup.find_all("h5"))
    print "******h5 processed end ********"

    print "******h6 processed ********"
    if(soup.find_all("h6")==[]):
	    h6_processed=[]
    else:
	    #h6_processed=process_text(soup.h6.text)
        h6_processed = process_array(soup.find_all("h6"))
    print "******h6 processed end ********"

    print "******a processed ********"
    if(soup.find_all("a")==[]):
	a_processed=[]
    else:
	#a_processed=process_text(soup.a.text)
        array1 = soup.find_all("a")
        #a_processed = [process_text(element.text) for element in array1]
        a_processed = process_array(array1)
	    #a_processed=[]
    #print soup.find_all("a")
	    #print soup.a.text
    print "******a processed end ********"

    print "body processed*************"
    if(soup.find_all("body")==[]):
	body_processed=[]
    else:
	body_processed=process_text(soup.body.text.encode('ascii','ignore'))
    print "body processed end*************"
    

    #Insert the processed document into mongodb
    #Do not change 
    webpages = mongodb_db.webpages
    document_to_insert = {
        "url": url,
        "title": title_processed,
        "h1": h1_processed,
        "h2": h2_processed,
        "h3": h3_processed,
        "h4": h4_processed,
        "h5": h5_processed,
        "h6": h6_processed,
        "a" : a_processed,
        "body": body_processed,
        "filename": file_name,
        "index": index
        }
    webpage_id = webpages.insert_one(document_to_insert)
    #####################Task t2c: your code #######################
   
    #Do not change below
    #Write the processed document
    new_file_name = file_name.replace("downloads/", "processed/")
    f = open("processedFileNamesToUUID.txt", "a")
    f.write(new_file_name + "|" + url + "\n")
    f.flush()
    f.close()

    f = open(new_file_name, "w")
    #body_processed.encode('ascii','ignore')
    f.write(body_processed.encode('ascii','ignore'))
    f.close()

#helper function for h tags and a tags
# use if needed
def process_array(array):
    processed_array = [process_text(element.text) for element in array]
    return processed_array

#This function does the necessary text processing of the text
def process_text(text):
    processed_text = ""
    #text=text.text
    #text = text1.get_text()
	
    #####################Task t2d: your code #######################
    #Given the text, do the following:
    #   convert it to lower case
    #   remove all stop words (English)
    #   remove all punctuation 
    #   stem them using Porter Stemmer
    #   Lemmatize it
    #####################Task t2d: your code #######################
    text=text.lower()
    #print text
    #word=set(stopwords.words('english'))
    stop=stopwords.words('english')
    #text=[i for i in text.split() if i not in stop]
    text= ' '.join(i for i in text.split() if i not in stop)
    #print text
    #text=[w for w in text if not w in word]
    #print text
    #toker = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
    #toker.tokenize(text)
    #regex = re.compile('[%s]' % re.escape(string.punctuation))
    #text=regex.sub('', text)
    #tokenizer = RegexpTokenizer(r'\w+')
    #tokenizer.tokenize(text)
    #print text
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    #a=text
    #text = a.translate(string.maketrans("",""), string.punctuation)
    #print text
    stemmer=PorterStemmer()
    text=(stemmer.stem(text))
    a = WordNetLemmatizer()
    text=a.lemmatize(text)
    #print text.encode('ascii','ignore')
    processed_text=text
	
    return processed_text 


#This function determines the vocabulary after processing
def find_corpus_vocabulary(file_name):
    vocabulary = []
    top_5000_words = None
    #Document frequency is a dictionary
    #   given a word, it will tell you how many documents this word was present it
    # use the variable document_frequency 
    document_frequency = defaultdict(int)
    #document_frequency = Counter()
    cnt = Counter()
    voc = []
    x = []
    dict1={}
    #####################Task t2e: your code #######################
    # The input is the file name with url and processed file names
    # for each file:
    #   get all the words and compute its frequency (over the entire corpus)
    # return the 5000 words with highest frequency
    # Hint: check out Counter class in Python
    #####################Task t2e: your code #######################
    with open(file_name) as myfile:
	for line in myfile:
		file = [item.strip() for item in line.split('|')]
		f = open(file[0])
	    	file_contents = f.read()
		words=word_tokenize(file_contents)
		vocabulary.extend(words)
		#voc.append(file_contents)
    		#f.close()
		
	#print "Voc list******"
	#print voc
	#print "Voc list end******"
	#vocabulary=''.join(voc)
	print "Joining*************"
	#words=word_tokenize(vocabulary)
	print "tokenizing*****"
	print "***********words**********"
        print vocabulary
        print "************words end***********"
	top_5000=dict(Counter(vocabulary).most_common(5000))
	top_5000_words=top_5000.keys()
	print "***********top 5000 words*************"
	sorted_top_5000_words = sorted(top_5000.items(), key=operator.itemgetter(1),reverse=True)
	print sorted_top_5000_words
	print "***********top 5000 ends**************"
	for word in top_5000_words:
		#doccount=0
		with open(file_name) as myfile1:
			for line in myfile1:
				file = [item.strip() for item in line.split('|')]
				f = open(file[0])
   				file_contents = f.read()
				if ((file_contents.split().count(word)) > 0):
       					document_frequency[word]+=1
			#print word, doccount
			#dict1[word]=doccount			
		
	#print dict1
	print "******document frequency***********"
	print document_frequency.items()
	print "******document frequency***********"	
	f = open("vocabulary.txt", "w")
	for word in top_5000_words:
        	f.write(word + "," + str(document_frequency[word]) + "\n")
    	f.close()

    return top_5000_words


def corpus_to_document_vec(vocabulary_file_name, file_name, output_file_name):
    #####################Task t2f: your code #######################
    # The input is the file names of vocabulary, and 
    #   the file  with url, processed file names  and the output file name
    #   the output is a file with tf-idf vector for each document
    #Pseudocode:
    # for each file:
    #   call the function text_to_vec with document body
    #   write the vector into output_file_name one line at a time
    #   into output_file_name
    #       ie document i will be in the i-th line
    #####################Task t2f: your code #######################
    processed=[]
    with open(file_name) as bdyfile:
            for line in bdyfile:
		tf_idf_vec=[]
                file = [item.strip() for item in line.split('|')]
		processed.append(file[0])
    
    for item in processed:
    	f = open(item)
       	body_contents = f.read()
	f.close()
       	tf_idf_vec=text_to_vec(vocabulary_file_name,body_contents)
	f1=open(output_file_name,"a")
	count=1
	for i in tf_idf_vec:
		#if(count<5000):
		f1.write(str(i)+" ")
		#count+=1
	f1.write("\n")
	f1.close()
       	#count+=1
			
    pass


def text_to_vec(vocabulary, text):
    #####################Task t2g: your code #######################
    # The input are vocabulary and text
    #   compute its tf-idf vector (ignore all words not in vocabulary)
    #Remember to use the variable document_frequency for computing idf
    #####################Task t2g: your code #######################
    tfidf = []
    vcb_list = []
    doc_frq_list = []
    #i = 0
    with open(vocabulary) as vocab:
        for line in vocab:
            vcb = [item.strip() for item in line.split(',')]
            vcb_list.append(vcb[0])
            doc_frq_list.append(vcb[1])
    count = 0
    while(count<=total_number_of_docs):			
    	for word in vcb_list:
        	#print word
        	term_freq = text.count(str(word))
        	doc_freq = float(doc_frq_list[count])
        	idf_score = 1+(math.log(float(total_number_of_docs)/doc_freq))
		#print idf_score
        	tf_idf = float(term_freq*idf_score)
		#tf_idf = tf_idf/numpy.linalg.norm(tf_idf)
        	tfidf.append(tf_idf)
		count+=1
        	#print word,tf_idf
    tfidf = tfidf/np.linalg.norm(tfidf)
    return tfidf


def query_document_similarity(query_vec, document_vec):
    #####################Task t2h: your code #######################
    #   Given a query and document vector
    #   compute their cosine similarity
    #cos=[]
    #query_vec_processed = process_text(query_vec)
    #print query_vec_processed
    #query_vector=text_to_vec("vocabulary.txt",query_vec_processed)
    #print "************qv***********"
    #print len(query_vector)
    #print len(document_vec)
    #q_vec=query_vec
    #doc_vec=np.array(np.loadtxt(document_vec))
    #print doc_vec.shape
    #for row in doc_vec:
    cosine_similarity=sk.metrics.pairwise.cosine_similarity(query_vec,document_vec)
    #print cosine_similarity
    #f = open("cosine_similarity.txt", "w")	
    #f.write(str(cosine_similarity))
    #f.close()
    #####################Task t2h: your code #######################
    return cosine_similarity

def rank_documents_tf_idf(query, k=10):
    #####################Task t2i: your code #######################

    #convert query to document using text_to_vec function
    q_processed=process_text(query)
    #print q_processed
    q_processed_token = (word_tokenize(str(q_processed)))
    if len(q_processed_token) >= 2:
        q_processed_token.append(q_processed)
    query_as_document = np.array(text_to_vec('vocabulary.txt',q_processed))
    print query_as_document, q_processed_token
    ranked_documents = None
    #Write code for the following:
    #   transform the query using process_text
    #   issue the transformed query to mongodb
    #   get ALL matching documents
    #   for each matching document:
    #       retrieve its tf-idf vector (use the file_name and index fields from mongodb)
    #   compute the tf-idf score and sort them accordingly 
    # return top-k documents
    #print mongodb_db.collection_names()
    #print processed_query
    #posts = mongodb_db.webpages
    #posts.insert(query)
    #print posts
    #print collection.find()
    #a=mongodb_db.webpages.find().distinct("filename")
    a = mongodb_db.webpages.find()
    '''for rec in a:
	    print (rec['filename']+','+rec['index'])'''
    #a=mongodb_db.webpages.find("filename")
    #print a
    #print list(mongodb_db.webpages.find({'index': {'$in': a }}))
    #print list(mongodb_db.webpages.find({'cse': ''}))
    #print collection.ensureIndex({ "content": "index" })
    #print collection.find({ "$index": { "$search": "cse" } })
		#print 'hello'
    #for webpage in mongodb_db.webpages.find(query):
	#print webpage
    	#for post in posts.find():
	#	print post
    dict_cos={}
    q_doc_vec=np.array(np.loadtxt("tf_idf_vector.txt"))
    for doc in a:
        #print doc['filename'],doc['index']
        #f=open(doc['filename'])
        #file_contents=f.read()
        for token in q_processed_token:
            if ((token in doc['body']) | (token in doc['title'])):
                q_ind = doc['index']
                q_tf_idf = q_doc_vec[q_ind]
                #print doc['filename'],q_ind,q_tf_idf
                cos= query_document_similarity(query_as_document,q_tf_idf)
    	        dict_cos[doc['index']]=cos
    print dict_cos
    sorted_items=sorted(dict_cos.items(), key=operator.itemgetter(1),reverse=True)
    print sorted_items
    ranked_documents = [ item[0] for item in sorted_items ]
    print ranked_documents[:k]
    #####################Task t2i: your code #######################
    return ranked_documents[:k]

def rank_documents_zone_scoring(query, k=10):
    #####################Task t2j: your code #######################

    #convert query to document using text_to_vec function
    q_processed=process_text(query)
    print q_processed
    q_tokenized=[]
    q_tokenized=word_tokenize(q_processed)
    print q_tokenized
    query_as_document = text_to_vec('vocabulary.txt',q_processed)
    print query_as_document
    ranked_documents = None
    #Write code for the following:
    #   transform the query using process_text
    #   issue the transformed query to mongodb
    #   get ALL matching documents
    #   for each matching document compute its score as following:
    #       score = 0
    #       for each word in query:
    #           find which "zone" the word fell in and give appropriate score
    #           title = 0.3, h1 = 0.2, h2=0.1, h3=h4=h5=h6=0.05,a: 0.1, body: 0.1
    #   so if a query keyword occurred in title, h1 and body, its score is 0.6
    #       compute this score for all keywords
    #       score of the document is the score of all keywords
    # return top-k documents
    zone_score={}
    a = mongodb_db.webpages.find()
    for doc in a:
    	if ((q_processed in doc['body']) | (q_processed in doc['title'])):
        	q_ind = doc['index']
        	score=0
                print doc['filename']
		for word in q_tokenized:
			if word in doc['body']:
				score +=0.1
			if word in doc['title']:
				score +=0.3
			if word in doc['h1']:
				score +=0.2
			if word in doc['h2']:
				score +=0.1
			if word in doc['h3']:
				score +=0.05
			if word in doc['h4']:
				score +=0.05
			if word in doc['h5']:
				score +=0.05
			if word in doc['h6']:
				score +=0.05
			if word in doc['a']:
				score +=0.1
			zone_score[doc['index']]=score
    print zone_score
    sorted_items=sorted(zone_score.items(), key=operator.itemgetter(1),reverse=True)
    ranked_documents = [ item[0] for item in sorted_items ]
    print ranked_documents[:k]

    #####################Task t2j: your code #######################
    #return ranked_documents[:k]

def rank_documents_pagerank(query, k=10):
    #####################Task t2k: your code #######################

    #convert query to document using text_to_vec function
    q_processed=process_text(query)
    print q_processed
    q_tokenized=[]
    q_tokenized=word_tokenize(q_processed)
    print q_tokenized
    query_as_document = text_to_vec('vocabulary.txt',q_processed)
    print query_as_document
    ranked_documents = None
    #Write code for the following:
    #   transform the query using process_text
    #   issue the transformed query to mongodb
    #   get ALL matching documents
    #   order the documents based on their pagerank score (computed in task 3)
    # return top-k documents
    a = mongodb_db.webpages.find()
    for doc in a:
    	if ((q_processed in doc['body']) | (q_processed in doc['title'])):
        	q_ind = doc['index']

    #####################Task t2k: your code #######################
    return ranked_documents[:k]


#Do not change below
def rank_documents(query):
    print "Ranking documents for query:", query
    print "Top-k for TF-IDF"
    print rank_documents_tf_idf(query)
    print "Top-k for Zone Score"
    print rank_documents_zone_scoring(query)
    print "Top-k for Page Rank"
    print rank_documents_pagerank(query)

setup_mongodb()
#####Uncomment the following functions as needed
process_document_corpus("fileNamesToUUID.txt")
vocabulary = find_corpus_vocabulary("processedFileNamesToUUID.txt")
corpus_to_document_vec("vocabulary.txt", "processedFileNamesToUUID.txt", "tf_idf_vector.txt")
query_document_similarity("Researching","tf_idf_vector.txt")
rank_documents_tf_idf("research interest", 10)
rank_documents_zone_scoring("faculty highlights")
