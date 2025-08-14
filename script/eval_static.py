#!/usr/bin/env python3
# -*- coding: UTF8 -*-
from __future__ import division
from operator import itemgetter, attrgetter, add
import numpy as np
#import gensim
import nltk
import sys
import os
import re
import csv
import string
import argparse
from scipy import spatial
import torch
import torch.jit
from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer, BertModel, DebertaTokenizer, DebertaModel, MPNetTokenizer
from transformers import AutoTokenizer, AutoModel
import random as rn
import math
#from InstructorEmbedding import INSTRUCTOR
#from adapters import AutoAdapterModel
#from simcse import SimCSE
#import specter
    
""" Author Amir Hazem
    date 15/03/2024
"""

""" Role
    This script evaluates static embedding representation methods
    for the alignment of patents with their corresponding products
    It builds aggregate word emebeddings for technology (patents)
    and market (companie's products)
    for English and Japanese data set
    Input:  test list of patent abstracts and products description
    Output: alignment score in Accuracy (acc@1 acc@5 acc@10 acc@100)
    and Mean average precision (Map)
"""



# Fix seeds to reproduce the same results #--
os.environ['PYTHONHASHSEED'] = '0'        #--
np.random.seed(42)                        #--
rn.seed(12345)                            #--   
# ----------------------------------------#--

os.environ["PYTORCH_USE_CUDA_DSA"] = "1"
os.environ['CUDA_LAUNCH_BLOCKING']="1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"




def load_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", help="w2v or sbert or e5", type=str, default="w2v")
    parser.add_argument("--train", action="store_true", help="generate training_data", default="False")
    parser.add_argument("--dev", action="store_true", help="generate training_data", default="False")
    parser.add_argument("--test", action="store_true", help="generate training_data", default="False")
    parser.add_argument("--dim", help="embedding dimension size", type=int, default="300")
    parser.add_argument("--norm", action="store_true", help="Normalize embedding")
    parser.add_argument("--average", action="store_true", help="Averaging patent embedding sum, default is the sum", default="False")
    parser.add_argument("--remove_stopwords", action="store_true", help="Remove stopwords", default="False")
    parser.add_argument("--pretrained", action="store_true", help="Use oretrained embeddings for w2v")
    parser.add_argument("--min_abstract_length", help="minimal abstract length in number of tokens", type=int, default="10")

    return (parser.parse_args())

def load_stop_words(path_stopwords):
    # load English stopwords
    stopwords = {}
    f = open(path_stopwords,'r')
    for line in f:
        str_ =line.split()
        stopwords[str_[0]]=str((str_[0].strip()).lower())
    f.close()
    return stopwords


def clean(text, stopwords):
    line = text.rstrip().lower()
    tokenized_text = ' '.join(nltk.word_tokenize(line))
    filter_stopwords = ' '.join(x for x in tokenized_text.split(' ') if x not in stopwords)
    return (filter_stopwords)

def clean_tokenize(text, stopwords):
    line = text.rstrip().lower()
    tokenized_text = ' '.join(nltk.word_tokenize(line))
    #filter_stopwords = ' '.join(x for x in tokenized_text.split(' ') if x not in stopwords)
    return (tokenized_text)

# Load pretrained word2vec
def load_pretrained_w2v(path_file):
    word2vec = gensim.models.KeyedVectors.load_word2vec_format(path_file, binary=True)
    return (word2vec)


# Load word2vec trained on technology data (patents)
def load_technology_w2v(path_file):
    embeddings = {}
    f = open(os.path.join(path_file))
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings[word] = coefs
    f.close()
    return (embeddings)

def load_train_test_set(file_directory, file_name):
    
    file = open(os.path.join(file_directory, file_name))
    csvreader = csv.reader(file, delimiter= ',')
    header = next(csvreader)
    org2patent = {}
    org2paper = {}
    test_set = {}
    patent2org = {}
    paper2org = {}
    for line in csvreader:
        
        if len(line) == 3:
            org = line[0]
            test_set[org] = org
            patent_list = line[1]
            paper_list = line[2]
            org2patent[org] = patent_list
            org2paper[org] = paper_list
            for i in patent_list.split(' '):
                if i in patent2org:
                    patent2org[i] = patent2org[i] + ' ' + org
                else:
                    patent2org[i] = org 

            for i in paper_list.split(' '):
                if i in paper2org:
                    paper2org[i] = paper2org[i] + ' ' + org
                else:
                    paper2org[i] = org        


    return (test_set, org2patent, org2paper, patent2org,paper2org)


def load_product_description(filename,full,test_list):

    org2desc = {}
    candidates = []
    reference_org2patent = {}
    file = open(filename)
    csvreader = csv.reader(file)
    header = next(csvreader)
    for line in csvreader:
        symbol = line[0]
        if symbol in test_list:
            list_patents = line[3]
            desc = line[full]
            #ch = symbol + '\t' + list_patents + '\t' + desc
            org2desc[symbol] = desc
            if len(desc) == 0 :
                print("size problem")

    return (org2desc)

def  load_patents(filename,list_):
    g_patent = {}
    file = open(filename)
    csvreader = csv.reader(file, delimiter= ',')
    header = next(csvreader)
    cpt = 0
    cpt_del = 0
    count_abstract = 0
    for line in csvreader:
        
        if len(line) == 2:
            patent_id = line[0]
            if patent_id in list_:
                abstract = line[1]
                cpt += 1
                
                if len(abstract.split(' ')) > 5:
                    count_abstract += 1
                    g_patent[patent_id] = abstract
                else:
                    cpt_del +=1     

    print("Size patents All " + str(cpt) )
    print("Size patents with abstract " + str(count_abstract) + '\t' + str(len(g_patent)))
    return(g_patent)

def  load_papers(filename,list_):
    g_paper = {}
    file = open(filename)
    csvreader = csv.reader(file, delimiter= ',')
    header = next(csvreader)
    cpt = 0
    cpt_del = 0
    count_abstract = 0
    for line in csvreader:
        if len(line) == 2:
            paper_id = line[0]
            abstract = line[1]
            if paper_id in list_:
                cpt += 1
                if len(abstract.split(' '))> 5:
                    count_abstract += 1
                    g_paper[paper_id] = abstract
                else:
                    cpt_del +=1     

    print("Size patents All " + str(cpt) )
    print("Size patents with abstract " + str(count_abstract) + '\t' + str(len(g_paper)))
    return(g_paper)


def get_symbol_from_url(page, directory):
    ch = page.split("___")[0]
    string = ch[len(directory):]
    return(string)

def load_product_web_pages(directory):
    web_pages_tmp = {}
    # iterate over files in
    # that directory
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            #print(f)
            web_pages_tmp[f] = f
    return(web_pages_tmp)

def load_product_webpages(product_directory, test):
    tab_vocab = {}
    print(test)
    web_pages = load_product_web_pages(product_directory)
    print("Total number of  webpages")
    print(len(web_pages))

    tab_found_symbols = {}
    product_description = {}
    for page in web_pages:
        #print(page)
        symbol = (get_symbol_from_url(page, product_directory)).rstrip()
        
        if symbol.strip() in test:
            print(symbol)
            if symbol not in tab_found_symbols and symbol not in sum_vect_embeddings_products:
                tab_found_symbols[symbol] = symbol
                        
                
            
            #print(symbol)
            path_file = page
            f = open(os.path.join(path_file))
            tab_words = {}
            for line in f:
                line = (' '.join(nltk.word_tokenize(line.strip().lower())))
                if len(line.strip()) > 0:

                    #print(line)
                    values = line.split(' ')
                    for w in values:
                        
                        if w not in stopwords and w not in string.punctuation:
                            #print(w)

                            tab_words[w] = w
            if symbol in  product_description:               
                product_description[symbol] = product_description[symbol] + ' ' + tab_words       
            else:

                product_description[symbol] = tab_words    
            f.close()
        
        
    return(product_description,tab_vocab)


def load_paper_patent_citations(tab_org):
    patent_citation_path = "../prepare_data/paperpatentcitation.csv"
    patent_to_paper, citation_pairs_all,citations_all = load_paper_citation_pairs(patent_citation_path,g_patent,g_paper)

    print(len(citation_pairs_all))

    print("total number of patent that cite papers")
    print(len(patent_to_paper))

    
    cpt_patent_that_cite_a_paper = 0
    tab_org2paper = {}
    tab_org2patent = {}
    paper2patent = {}
    patent2paper = {}
    tab_patent_that_cite_papers = {}
    for i in tab_org:

        ch =  tab_org[i].split(' ')
        for patent in ch:
            if patent in patent_to_paper:
                cpt_patent_that_cite_a_paper +=1
                tab_patent_that_cite_papers[patent] = patent


                    
                if i in tab_org2paper:
                    tab_org2paper[i] = tab_org2paper[i] + ' ' + patent_to_paper[patent]
                    tab_org2patent[i] = tab_org2patent[i] + ' ' + patent 
                else:
                    tab_org2paper[i] = patent_to_paper[patent] 
                    tab_org2patent[i] = patent
                        
                for j in patent_to_paper[patent].split(' '):
                    if j not in paper2patent: 
                        paper2patent[j] = patent
                    else:
                        paper2patent[j] = paper2patent[j] + ' ' + patent

                    if patent in patent2paper:
                        patent2paper[patent] = patent2paper[patent] + ' ' + j
                    else:   
                        patent2paper[patent] =  j

    print(cpt_patent_that_cite_a_paper)
    print(len(tab_patent_that_cite_papers))
    print("number of organization that are citing papers")  
    print(len(tab_org2paper))

    print(len(paper2patent))
    return(patent2paper, paper2patent)   

def load_paper_citation_pairs(patent_citation_path,g_patent,g_paper):

    file = open(patent_citation_path)
    csvreader = csv.reader(file, delimiter= ',')
    header = next(csvreader)

    citation_pairs = {}
    citations = {}
    patent_to_paper = {}
    for line in csvreader:
            #print(line)
            #print(len(line))
        if len(line) == 2:

            
            citation_id = line[1] # is the patent_id
            paper_id = line[0]
            if paper_id in g_paper and citation_id in g_patent:

                #citations[patent_id] = patent_id
                citations[citation_id] = citation_id

                if citation_id in patent_to_paper:

                    patent_to_paper[citation_id] = patent_to_paper[citation_id] + ' ' + paper_id
                    
                else:
                    patent_to_paper[citation_id] =  paper_id

                if  paper_id in citation_pairs:
                    citation_pairs[paper_id] = citation_pairs[paper_id] + ' ' + citation_id
                
                else:
                    citation_pairs[paper_id] = citation_id

    return(patent_to_paper, citation_pairs,citations)

def sentence_average_embeddings(data, embedding_index, stopwords, dim, average, stopwords_bool):
    sum_vect_embeddings = {}
    tab_abst = {}
    abst = ""
    
    for _id in data:
        
        if stopwords_bool:
            abst = clean(data[_id],stopwords)
        else:
            abst =  data[_id]
        
        tab_abst[_id] = abst
        sum_vect_embeddings[_id]= np.zeros(dim,float)
        desc = abst.split(' ')
        cpt_token = 0

        for token in desc:
            if token in embedding_index:
                cpt_token += 1
                sum_vect_embeddings[_id] = np.add(sum_vect_embeddings[_id],embedding_index[token])

        if average:
            sum_vect_embeddings[_id] = [x / cpt_token for x in sum_vect_embeddings[_id]]
    return(sum_vect_embeddings,tab_abst)

def sum_embeddings_abst_aggregate(reference_org2patent_all, sum_vect_patent_embeddings, dim, average):

    sum_vect_embeddings_tmp = {}
    for symbol in reference_org2patent_all:
        

        patent_symbol = reference_org2patent_all[symbol].split(' ') 


        sum_vect_embeddings_tmp[symbol]= np.zeros(dim,float)
        cpt_pat = 0
        for patent_id  in patent_symbol:
            if patent_id in sum_vect_patent_embeddings:
                sum_vect_embeddings_tmp[symbol] = np.add(sum_vect_embeddings_tmp[symbol],sum_vect_patent_embeddings[patent_id])
                cpt_pat += 1
        if average:
        # Normalize 
            sum_vect_embeddings_tmp[symbol] = [x / cpt_pat for x in sum_vect_embeddings_tmp[symbol]]

        if np.isnan(sum_vect_embeddings_tmp[symbol]).any():
            print(symbol)
        
    return(sum_vect_embeddings_tmp)


def combine_abst_aggregate(org2patent, sum_vect_embeddings_patents, sum_vect_embeddings_papers,dim,average):

    sum_vect_embeddings_tmp = {}
    for symbol in org2patent:
        
        sum_vect_embeddings_tmp[symbol]= np.zeros(dim,float)
         
        sum_vect_embeddings_tmp[symbol] = np.add(sum_vect_embeddings_patents[symbol],sum_vect_embeddings_papers[symbol])
        
        if average:
        # Normalize 
            sum_vect_embeddings_tmp[symbol] = [x / 2 for x in sum_vect_embeddings_tmp[symbol]]

        if np.isnan(sum_vect_embeddings_tmp[symbol]).any():
            print(symbol)
        
    return(sum_vect_embeddings_tmp)

def compute_scores(mean_patent_embeddings_test, sum_vect_embeddings_products):


    map_ = 0
    test_size = len(sum_vect_embeddings_products)
    print("test size = \t" + str(test_size))
    #test_size = 100 # for a matter of comparison with bert
    tab_precisions = [1, 5, 10, 100, test_size]
    map_ = 0
    tab_acc = {}
    cpt_error = 0
    for i in tab_precisions:
        tab_acc[i] = 0

    for patent_src in  mean_patent_embeddings_test:

        tab_res = []    

        for product_cand  in sum_vect_embeddings_products:
            cosine = (1- spatial.distance.cosine(mean_patent_embeddings_test[patent_src], sum_vect_embeddings_products[product_cand]))  
            #print(len(mean_patent_embeddings_test[patent_src]))
            #print(len(sum_vect_embeddings_products[product_cand]))
            #print("----")
            if cosine != 1:
                tab_res.append((product_cand,cosine))
            else:
                cpt_error +=1
                #print("Duplicates maybe present in the test set...")
                #print(patent_src + '\t' + product_cand)
                #print(mean_patent_embeddings_test[patent_src])
                #print(sum_vect_embeddings_products[product_cand])
                #print(np.all(mean_patent_embeddings_test[patent_src]==0))
                #sys.exit()

        result = sorted(tab_res,key=itemgetter(1),reverse=True )

        map_tmp = 0     
        tab_acc_tmp = {}

        for i in tab_precisions:
            tab_acc_tmp[i] = 0
        
        cpt = 0
        acc_tmp = 0
        for cand in result:
            cpt += 1
            if cand[0] == patent_src:

                acc_tmp += 1
                #print(patent_src+ ' found \t' + cand[0] + '\t'+ str(cand[1])+ '\t' + str(cpt))
                if (cpt<=100):
                    map_tmp += acc_tmp/cpt      
                for precision_at_n in tab_acc_tmp:
                    if cpt <= precision_at_n  :
                        tab_acc_tmp[precision_at_n] += 1    

        for i in tab_acc_tmp:
            tab_acc[i] += tab_acc_tmp[i]


        map_+= map_tmp


    # Print scores
    print(len(mean_patent_embeddings_test))

    res_map = (map_/ (len((mean_patent_embeddings_test))))*100
    score = {}
    for i in tab_acc:
        score[i] = (tab_acc[i] / test_size)*100

    accuracy= (tab_acc[test_size] / (len(mean_patent_embeddings_test))) * 100
    accuracy100 = (tab_acc[100] / (len(mean_patent_embeddings_test))) * 100
    accuracy10 = (tab_acc[10] / (len(mean_patent_embeddings_test))) * 100
    accuracy5 = (tab_acc[5] / (len(mean_patent_embeddings_test))) * 100
    accuracy1 = (tab_acc[1] / (len(mean_patent_embeddings_test))) * 100
    print(str(round(accuracy1,2)) + " & " + str(round(accuracy5,2)) + " & " + str(round(accuracy10,2)) + " & " + str(round(accuracy100,2)) + " & " + str(round(accuracy,2)) + " & " + str(round(res_map,2)) )
    print(str(round(score[1],2)) + " & " + str(round(score[5],2)) + " & " + str(round(score[10],2)) + " & " + str(round(score[100],2)) + " & " + str(round(score[test_size],2)) + " & " + str(round(res_map,2)) )


def sum_embeddings_BERT(patent_abstract, stopwords, dim):
    
    #device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device('cuda')

    #inputs    = tokenizer(sentence, return_tensors="pt").to(device)


    # Load pre-trained model tokenizer (vocabulary)
    #tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    #model = BertModel.from_pretrained('bert-base-uncased',output_hidden_states = True)


    #tokenizer = MPNetTokenizer.from_pretrained('microsoft/mpnet-base')
    #model = MPNetTokenizer.from_pretrained('microsoft/mpnet-base',output_hidden_states = True)


    #from fairseq.models.masked_permutation_net import MPNet
    #model = MPNet.from_pretrained("/home/andrea/Desktop/work/rcast/year_2024_2025/conferences/mpnet/MPNet/pretraining/checkpoints/checkpoint46/", "checkpoint_best.pt", bpe='bert')
    #mpnet = MPNet.from_pretrained('checkpoints', "/home/andrea/Desktop/work/rcast/year_2024_2025/conferences/mpnet/MPNet/pretraining/checkpoints/checkpoint46/", 'path/to/data', bpe='bert')
    # The same interface can be used with custom models as well
    from fairseq.models.transformer_lm import TransformerLanguageModel
    model = TransformerLanguageModel.from_pretrained("/home/andrea/Desktop/work/rcast/year_2024_2025/conferences/mpnet/MPNet/pretraining/checkpoints/", "checkpoint_best.pt", bpe='bert')
    assert isinstance(model, torch.nn.Module)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    src_text="this is a test" # input text, e.g., a PubMed abstract
    tokenized_text = model.encode(src_text)
    print(tokenized_text)    
    
    segments_ids = [1] * len(tokenized_text)
    
    segments_tensors = torch.tensor([segments_ids]).to(device)
    print(segments_tensors)
    with torch.no_grad():
        outputs = model(tokenized_text)
    sys.exit()


    # Define the model repo
 

    #mpnet = MPNet.from_pretrained('checkpoints', 'checkpoint_best.pt', '/home/andrea/Desktop/work/rcast/year_2024_2025/conferences/mpnet/MPNet/pretraining/checkpoints/checkpoint46/', bpe='bert')
    #assert isinstance(mpnet.model, torch.nn.Module)
    #from transformers import AutoModel, AutoTokenizer 
    #model_name = "/home/andrea/Desktop/work/rcast/year_2024_2025/conferences/mpnet/MPNet/pretraining/checkpoints/checkpoint46/"
    #model = torch.jit.load('/home/andrea/Desktop/work/rcast/year_2024_2025/conferences/mpnet/MPNet/pretraining/checkpoints/checkpoint46/checkpoint_best.pt')
     


    #model = AutoModel.from_pretrained(model_name)
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    #sys.exit()

    # Download pytorch model
    #.from transformers import AutoModel, AutoTokenizer 
    #.model_name = "microsoft/mpnet-base"
     
    #.model = AutoModel.from_pretrained(model_name)
    #.tokenizer = AutoTokenizer.from_pretrained(model_name)



    #from transformers import XLNetTokenizer, XLNetModel

    #tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
    #model = XLNetModel.from_pretrained('xlnet-base-cased')


    
    #tokenizer = DebertaTokenizer.from_pretrained('microsoft/deberta-base')
    #model = DebertaModel.from_pretrained('microsoft/deberta-base',output_hidden_states = True)
    #tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
    #model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
    
    model     = model.to(device)
    #from transformers import LongformerModel, AutoTokenizer
    #model = LongformerModel.from_pretrained("allenai/longformer-base-4096")
    #tokenizer = AutoTokenizer.from_pretrained("allenai/longformer-base-4096")


    #from transformers import ReformerTokenizer, ReformerModel

    #tokenizer = ReformerTokenizer.from_pretrained('google/reformer-crime-and-punishment')
    #model = ReformerModel.from_pretrained('google/reformer-crime-and-punishment')

    #inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
    #outputs = model(**inputs)

    #last_hidden_states = outputs[0]  

    #tokenizer = BertTokenizer.from_pretrained('roberta-base')
    #model = BertModel.from_pretrained('roberta-base',output_hidden_states = True)


    #tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
    #model = BertModel.from_pretrained('bert-large-uncased',output_hidden_states = True)

    #tokenizer = BertTokenizer.from_pretrained('roberta-base')
    #model = BertModel.from_pretrained('roberta-base',output_hidden_states = True)
    #from transformers import AutoTokenizer, RobertaModel
    #import torch

    #tokenizer = AutoTokenizer.from_pretrained("roberta-base")

    #model = RobertaModel.from_pretrained("roberta-base")   

    sum_vect_embeddings = {}
    tab_abst = {}

    tab = {}
    abst= ""
    #print(patent_abstract)
    for i in patent_abstract:
        #print(i)
        #abst = clean(patent_abstract[i],stopwords)
        abst = patent_abstract[i]

        #print(len(abst.split(' ')))
        len_ = len(abst.split(' '))
        #if len_ >= 512:
        #   print(abst)
        #   abst = abst[:512]
        #   print(abst)
        #   sys.exit()          
        # Add the special tokens.
        marked_text = "[CLS] " + abst + " [SEP]"
        marked_text = abst 
        # Split the sentence into tokens.
        tokenized_text = tokenizer.tokenize(marked_text)
        #tokenized_text = tokenized_text.to(device)
        print(marked_text)
        print("__________________")
        print((tokenized_text))
        

        src_tokens = model.encode(abst)
        
        outputs = model(src_tokens)
        print(outputs)
        sys.exit()
        #output = model.decode(inputs)
        #print(output)
        generate = model.generate(src_tokens, beam=5)[0]
        output = model.decode(generate[0]["tokens"])
        print(output)
        #last_hidden_states = output.last_hidden_state
        #print(last_hidden_states)
        print("-------")
        sys.exit()
        if len(tokenized_text) > 512:
            tokenized_text = tokenized_text[:512] 
        # Map the token strings to their vocabulary indeces.
        indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

        segments_ids = [1] * len(tokenized_text)


        tokens_tensor = torch.tensor([indexed_tokens]).to(device)
        segments_tensors = torch.tensor([segments_ids]).to(device)

        print(tokens_tensor)
        sys.exit()
        print(segments_tensors)
        with torch.no_grad():

            outputs = model(tokens_tensor, segments_tensors)
            #print(outputs)
            #print(outputs.last_hidden_state)
            
            #print(outputs.last_hidden_state[0][0])
            #print(outputs.last_hidden_state[0][1])
            #print(outputs.last_hidden_state[0][12])
            #hidden_states = outputs[2]
            #print(hidden_states)
            #print(hidden_states[-1][0])
            #sys.exit()
            #print(outputs.hidden_states[0])
            #print(outputs.hidden_states[11])
            #print(outputs.hidden_states[12])
            
            
            #token_vecs = hidden_states[-2][0]# from 2 to 12
            #token_vecs = hidden_states[10][0]
            
            #.hidden_states = outputs[2] # for BERT
            hidden_states = outputs[0]  # for mpnET
            #print(hidden_states)
            #sys.exit()
            #hidden_states = outputs[1]
            #print(outputs)
            #print(hidden_states)
            
            token_vecs = hidden_states[-1][0]
            #print(token_vecs)
               
            #.sentence_embedding = torch.sum(token_vecs, dim=0) # for BERT
            sentence_embedding = token_vecs # for MPNet 
            #print(sentence_embedding)
             
            #sentence_embedding = hidden_states
            #print(token_vecs)
            #print(sentence_embedding.cpu().detach().numpy())
            #sys.exit()
            #hidden_states = outputs.hidden_states#[12]
            #print(outputs.hidden_states)
            #token_vecs = hidden_states[-1][0]
            #print(token_vecs)
            
            #sentence_embedding = torch.sum(token_vecs, dim=0)

            #print(sentence_embedding)
            #sys.exit()
            #print(sentence_embedding)
            #print(len(sentence_embedding))
            
            #print(abst)
            _id = i
            tab_abst[_id] = abst
        
            sum_vect_embeddings[_id]= np.zeros(dim,float)

            
            sum_vect_embeddings[_id] = sentence_embedding.cpu().detach().numpy()
            #print(sum_vect_embeddings[_id])
            
            #print(sum_vect_embeddings[_id])
            #sys.exit()
    #print(len(sum_vect_embeddings))
    #print(len(tab_abst))

    return(sum_vect_embeddings,tab_abst)    

#.import lightning
#.from lightning.pytorch import Trainer, LightningModule
#.class MyLightningModule(LightningModule):
#.    def __init__(self, learning_rate, *args, **kwargs):
#.        super().__init__()
#.        self.save_hyperparameters()

def sum_embeddings_SPECTER(patent_abstract, stopwords, stopwords_bool, model_type):
    
    # load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter')
    model = AutoModel.from_pretrained('allenai/specter')

    #tokenizer = AutoTokenizer.from_pretrained('/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/model-output/model')
    #model = AutoModel.from_pretrained('/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/model-output/')
    model_path = '/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/model-output/best.th'
    #model= torch.load(model_path)
   # import ModelClass
    #from specter.model  import Specter
    from allennlp.models.archival import load_archive
    from allennlp.models.model import Model
    from allennlp import __version__
    from allennlp.commands import ArgumentParserWithDefaults
    from allennlp.commands.predict import _PredictManager
    from allennlp.commands.subcommand import Subcommand
    from allennlp.common import Params, Tqdm
    from allennlp.common.checks import check_for_gpu, ConfigurationError
    from allennlp.common.params import parse_overrides
    #from allennlp.common.util import lazy_groups_of, import_submodules
    from allennlp.models import Archive
    from allennlp.models.archival import load_archive
    from allennlp.predictors.predictor import Predictor, JsonDict#, DEFAULT_PREDICTORS
    from allennlp.data import Instance, DatasetReader
    from overrides import overrides
    from tqdm import tqdm
    from allennlp.predictors import Predictor

    #archive = load_archive('/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/model-output/model.tar.gz')
    #model = Model.load('/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/model-output/config.json','/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/model-output/')
    #.model = Model.load('/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/model-output/','/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/model-output/')
    #from allennlp.training import checkpointer

    #model = load_archive.extract_module('/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/model-output/model.tar.gz')
    #model = from_archive(archive, self.predictor_name)
    model = Predictor.from_archive('config.json','/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/model-output/model.tar.gz')
    
     # type: ignore

    #model.load_state_dict(torch.load("/path/to/model/weights.th"))



    #model = AutoModel.from_pretrained('/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/model-output') 
    #model = AutoModel.from_pretrained('/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/new/')
    #from transformers.trainer_utils import EvalPrediction, get_last_checkpoint
    #from pytorch_lightning.callbacks import ModelCheckpoint
    #last_checkpoint = get_last_checkpoint('/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/model-output/') 
    #model = ModelClass.load_from_checkpoint('/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/model-output/')

    #model = MyLightningModule.load_from_checkpoint('/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/model-output')
    #import allennlp_models
    #from allennlp.predictors.predictor import Predictor

    #predictor = Predictor.from_path("/home/andrea/Desktop/work/rcast/conferences/coling2025/data/specter/data/model-output")

    #print(last_checkpoint)
    #model = trainer.train(resume_from_checkpoint=last_checkpoint)
    sys.exit()
    sum_vect_embeddings = {}
    tab_abst = {}

    tab = {}
    abst= ""
    #print(patent_abstract)
    for _id in patent_abstract:
        #print(i)
        if stopwords_bool:
            abst = clean(patent_abstract[_id],stopwords)
        else:   
            abst = clean_tokenize(patent_abstract[_id],stopwords)
        
        #tab_abst[_id] = abst
        sum_vect_embeddings[_id]= np.zeros(dim,float)

        # preprocess the input
        inputs = tokenizer(abst, padding=True, truncation=True,return_tensors="pt", return_token_type_ids=False, max_length=512)
        output = model(**inputs)
        # take the first token in the batch as the embedding
        embeddings1 = output.last_hidden_state[:, 0, :]
        #print( embeddings1.cpu().detach().numpy()[0])

        
        #embeddings1 = model.encode(abst, convert_to_tensor=True)
        #print(embeddings1)
        #sum_vect_embeddings[_id] = embeddings1.detach().numpy()
        sum_vect_embeddings[_id] = embeddings1.cpu().detach().numpy()[0]

    return(sum_vect_embeddings,tab_abst)

def sum_embeddings_SPECTER2(patent_abstract, stopwords, stopwords_bool, model_type):
    
    #load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')

    #load base model
    model = AutoAdapterModel.from_pretrained('allenai/specter2_base')

    #load the adapter(s) as per the required task, provide an identifier for the adapter in load_as argument and activate it
    model.load_adapter("allenai/specter2", source="hf", load_as="specter2", set_active=True)

    sum_vect_embeddings = {}
    tab_abst = {}

    tab = {}
    abst= ""
    #print(patent_abstract)
    for _id in patent_abstract:
        #print(i)
        if stopwords_bool:
            abst = clean(patent_abstract[_id],stopwords)
        else:   
            abst = clean_tokenize(patent_abstract[_id],stopwords)
        
        #tab_abst[_id] = abst
        sum_vect_embeddings[_id]= np.zeros(dim,float)

        # preprocess the input
        inputs = tokenizer(abst, padding=True, truncation=True,return_tensors="pt", return_token_type_ids=False, max_length=512)
        output = model(**inputs)
        # take the first token in the batch as the embedding
        embeddings1 = output.last_hidden_state[:, 0, :]
        #print( embeddings1.cpu().detach().numpy()[0])

        
        #embeddings1 = model.encode(abst, convert_to_tensor=True)
        #print(embeddings1)
        #sum_vect_embeddings[_id] = embeddings1.detach().numpy()
        sum_vect_embeddings[_id] = embeddings1.cpu().detach().numpy()[0]

    return(sum_vect_embeddings,tab_abst)    


def sum_embeddings_SimCSE(patent_abstract, stopwords, stopwords_bool, model_type):
    

    # Import our models. The package will take care of downloading the models automatically
    #tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")
    #model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-base-uncased")


    #tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-bert-base-uncased")
    #model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-bert-base-uncased")
   

    #tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-bert-large-uncased")
    #model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-bert-large-uncased")
    #tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-bert-large-uncased")
    #model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-bert-large-uncased")

    tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-base")
    model     =     AutoModel.from_pretrained("princeton-nlp/unsup-simcse-roberta-base")

    #tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/unsup-simcse-roberta-large")
    #model = AutoModel.from_pretrained("princeton-nlp/unsup-simcse-roberta-large")

    #tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
    #model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-base")
    #tokenizer = AutoTokenizer.from_pretrained("princeton-nlp/sup-simcse-roberta-large")
    #model = AutoModel.from_pretrained("princeton-nlp/sup-simcse-roberta-large")

    
    
    #dim = 1024
    dim = 768
    #model = SimCSE("princeton-nlp/sup-simcse-bert-base-uncased")
    
    #model = SentenceTransformer('princeton-nlp/sup-simcse-bert-base-uncased',  device='cuda')

    model.max_seq_length = 512

    sum_vect_embeddings = {}
    tab_abst = {}

    tab = {}
    abst= ""
    #print(patent_abstract)
    for _id in patent_abstract:
        #print(i)
        if stopwords_bool:
            abst = clean(patent_abstract[_id],stopwords)
        else:   
            abst = clean_tokenize(patent_abstract[_id],stopwords)
        
        tab_abst[_id] = abst
        sum_vect_embeddings[_id]= np.zeros(dim,float)

        inputs = tokenizer(abst, padding=True, truncation=True, return_tensors="pt")
        
        # Get the embeddings
        with torch.no_grad():
            embeddings1 = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
        
        #embeddings1 = model.encode(abst, convert_to_tensor=True)
        #embeddings1 = model.encode(abst)
        #print(embeddings1.cpu().detach().numpy()[0])
        #print(embeddings2)
        #sys.exit()
        #sum_vect_embeddings[_id] = embeddings1.detach().numpy()
        sum_vect_embeddings[_id] = embeddings1.cpu().detach().numpy()[0]
        #print(sum_vect_embeddings[_id])
        
        #print(embeddings1.detach().numpy())
        

    return(sum_vect_embeddings,tab_abst) 

def sum_embeddings_SBERT(patent_abstract, stopwords, stopwords_bool, model_type):
    
    if model_type == "minio":
        dim = 384
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')

    else:
        
        #dim = 768
        #model = SentenceTransformer('all-mpnet-base-v2', device='cuda')
        dim = 384
        model = SentenceTransformer('all-MiniLM-L6-v2', device='cuda')
        #model = SentenceTransformer("allenai-specter", device='cuda')


        #model = SentenceTransformer('princeton-nlp/sup-simcse-bert-base-uncased',  device='cuda')
        #model = SentenceTransformer('princeton-nlp/unsup-simcse-bert-base-uncased',  device='cuda')
        


        #dim = 1024
        #model = SentenceTransformer('princeton-nlp/sup-simcse-bert-large-uncased',  device='cuda')


        #model = SentenceTransformer('princeton-nlp/sup-simcse-roberta-base',  device='cuda')
        #model = SentenceTransformer('AI-Growth-Lab/PatentSBERTa',  device='cuda')
        #model = SentenceTransformer('allenai/scibert_scivocab_uncased',  device='cuda')
        #model = SentenceTransformer('colbert-ir/colbertv2.0',  device='cuda')

        #dim = 1024
        #model = SentenceTransformer('Yanhao/simcse-bert-for-patent',  device='cuda')
        #model = SentenceTransformer('anferico/bert-for-patents',  device='cuda')




        #model = SentenceTransformer('sentence-transformers/sentence-t5-base',  device='cuda')
        #model = SentenceTransformer('sentence-transformers/gtr-t5-base',  device='cuda')
        #.model = SentenceTransformer('sentence-transformers/stsb-roberta-base',  device='cuda')
        #model = SentenceTransformer('sentence-transformers/nli-roberta-base',  device='cuda')
        #model = SentenceTransformer('sentence-transformers/stsb-roberta-base-v2',  device='cuda')
        #model = SentenceTransformer('sentence-transformers/stsb-bert-base',  device='cuda')
        #dim = 1024
        #model = SentenceTransformer('sentence-transformers/stsb-bert-large',  device='cuda')
        #model = SentenceTransformer('sentence-transformers/stsb-roberta-large',  device='cuda')
        #model = SentenceTransformer('sentence-transformers/sentence-t5-large',  device='cuda')
        #model = SentenceTransformer('sentence-transformers/sentence-t5-xl',  device='cuda')
        #model = SentenceTransformer('allenai/scibert_scivocab_uncased',  device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/base/config.json',  device='cuda')
        
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/base2/checkpoint-400',device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/base',device='cuda')
        #dim = 384
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/all-mpnet-base-v2-tech2market',device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/all-MiniLM-L6-v2-tech2market/checkpoint-4500/',device='cuda')
        #model = SentenceTransformer('microsoft/mpnet-base', device='cuda')
        #model = SentenceTransformer('microsoft/MiniLM-L12-H384-uncased', device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/mpnet-base-tech2market/', device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/minilm-base-tech2market/', device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/all-minilm-base-tech2market', device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/all-minilm-base-tech2market/checkpoint-1200/', device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/all-mpnet-base-v2-tech2market/checkpoint-750',device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/all-minilm-base-tech2market-cf2',device='cuda')
        #dim = 384
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/all-minilm-base-tech2market-cf0',device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/bert-minilm-tech2market-cf2/',device='cuda')
        #dim = 768

        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/bert-base-tech2market-cf1',device='cuda')
        
        
        #model = SentenceTransformer('sentence-transformers/stsb-bert-base',  device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/bert-base-tech2market-cf0',device='cuda')
        
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/bert-base-tech2market-cf2',device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/bert-mpnet-tech2market-cf2/',device='cuda')

        #model = SentenceTransformer('sentence-transformers/stsb-roberta-base-v2',  device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/roberta-base-tech2market-cf0',device='cuda')

        #dim = 1024
        #model = SentenceTransformer('sentence-transformers/stsb-bert-large',  device='cuda')

        # FInal proposed models
        # miniLM
        #dim = 384
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/bert-MiniLM-proposed-MultipleNegativesRankingLoss',device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/bert-MiniLM-proposed-Cosine/',device='cuda')
        
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/bert-base-proposed-MultipleNegativesRankingLoss-paper/',device='cuda')

        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/bert-base-proposed-MultipleNegativesRankingLoss-product/',device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/bert-base-proposed-MultipleNegativesRankingLoss-patent/',device='cuda')
        #model = SentenceTransformer('//home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/bert-base-proposed-MultipleNegativesRankingLoss-batch32-all/checkpoint-800/',device='cuda')
        #dim = 768
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/bert-allmpnet-proposed-MultipleNegativesRankingLoss/',device='cuda')

        
        # bert-base
        #model = SentenceTransformer('sentence-transformers/stsb-bert-base',  device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/bert-base-proposed-MultipleNegativesRankingLoss/',device='cuda')

        # e5
        # run e5 model not here for the baseline
        #model = SentenceTransformer('intfloat/e5-base-v2', device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/e5-base-proposed-MultipleNegativesRankingLoss/',device='cuda')

        # roberta-base
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/roberta-base-proposed-MultipleNegativesRankingLoss/',device='cuda') 
        
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/PatentSBERTa-base-proposed-MultipleNegativesRankingLoss',device='cuda')
        #dim = 384
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/new-bert-mini-baseline-MultipleNegativesRankingLoss/',device='cuda')
        #model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/new-bert-mini-baseline-TripletLossAmir/',device='cuda')
        #.model = SentenceTransformer('/home/andrea/Desktop/work/rcast/conferences/coling2025/script/data/sbert/model/new-bert-mini-baseline-TripletLoss-margin-0/',device='cuda')
        

        

    model.max_seq_length = 512

    sum_vect_embeddings = {}
    tab_abst = {}

    tab = {}
    abst= ""
    #print(patent_abstract)
    for _id in patent_abstract:
        #print(i)
        if stopwords_bool:
            abst = clean(patent_abstract[_id],stopwords)
        else:   
            abst = clean_tokenize(patent_abstract[_id],stopwords)
        
        tab_abst[_id] = abst
        sum_vect_embeddings[_id]= np.zeros(dim,float)
        embeddings1 = model.encode(abst, convert_to_tensor=True)
        #print(embeddings1)
        #sum_vect_embeddings[_id] = embeddings1.detach().numpy()
        sum_vect_embeddings[_id] = embeddings1.cpu().detach().numpy()
        #print(sum_vect_embeddings[_id])
        
        #print(embeddings1.detach().numpy())
        

    return(sum_vect_embeddings,tab_abst)



def sum_embeddings_Instructor(patent_abstract, stopwords, stopwords_bool, model_type, text_type):
    
    
    #model = INSTRUCTOR('hkunlp/instructor-base')
    #model = INSTRUCTOR('hkunlp/instructor-large')
    model = INSTRUCTOR('hkunlp/instructor-xl')
    #sentence = "3D ActionSLAM: wearable person tracking in multi-floor environments"
    #instruction = "Represent the Science title:"
    instruction = "Represent the " + str(text_type)+":"
    #instruction = "Represent the Science sentence:"

    model.max_seq_length = 512

    sum_vect_embeddings = {}
    tab_abst = {}

    tab = {}
    abst= ""
    #print(patent_abstract)
    for _id in patent_abstract:
        #print(i)
        if stopwords_bool:
            abst = clean(patent_abstract[_id],stopwords)
        else:   
            abst = clean_tokenize(patent_abstract[_id],stopwords)
        
        tab_abst[_id] = abst
        sum_vect_embeddings[_id]= np.zeros(dim,float)
        
        embeddings1 = model.encode([[instruction,abst]])
        #print(embeddings1[0])
        
        sum_vect_embeddings[_id] = embeddings1[0]
        
        #sum_vect_embeddings[_id] = embeddings1.cpu().detach().numpy()
        

    return(sum_vect_embeddings,tab_abst)


def mistral_e5(patent_abstract,stopwords,stopwords_bool):

    print("Mistral")    
    dim = 768
    dim = 1024
    model = SentenceTransformer('intfloat/e5-base-v2')
    #model = SentenceTransformer('intfloat/e5-large-v2')
    #model = SentenceTransformer('../data/sbert/model/e5/mye5model')
    #model = SentenceTransformer('intfloat/e5-mistral-7b-instruct')
    #model = SentenceTransformer('mistralai/Mistral-7B-v0.1')

    model.cuda()
    sum_vect_embeddings = {}
    tab_abst = {}

    tab = {}
    abst= ""
    #print(patent_abstract)
    for _id in patent_abstract:
        #print(i)
        if stopwords_bool:
            abst = clean(patent_abstract[_id],stopwords)
        else:
            abst =   patent_abstract[_id]
        
        input_texts = ["query: " + abst]
         
        #print(input_texts)
            
        #print(abst)
        
        tab_abst[_id] = abst
        sum_vect_embeddings[_id] = np.zeros(dim,float)
        #embeddings1 = model.encode(input_texts, normalize_embeddings=True)
        
        embeddings1 = model.encode(input_texts, normalize_embeddings=False)
        #print(embeddings1[0])
        sum_vect_embeddings[_id] = embeddings1[0]
        

    return (sum_vect_embeddings, tab_abst)


def sum_embeddings_SGPT(patent_abstract, stopwords, stopwords_bool, model_type):
    dim = 2048
    dim = 768
    device = torch.device('cuda')
    
    #model = SentenceTransformer("Muennighoff/SGPT-125M-mean-nli") # results lower than weighted mean bitfit
    #model = SentenceTransformer("Muennighoff/SGPT-125M-lasttoken-nli") --> does not work

    model = SentenceTransformer("Muennighoff/SGPT-125M-weightedmean-nli-bitfit")
    #model = SentenceTransformer("Muennighoff/SGPT-1.3B-weightedmean-nli-bitfit")
    #model = SentenceTransformer("Muennighoff/SGPT-2.7B-weightedmean-nli-bitfit")
    #model = SentenceTransformer("Muennighoff/SGPT-5.8B-weightedmean-nli-bitfit")
    model.cuda()

    sum_vect_embeddings = {}
    tab_abst = {}

    tab = {}
    abst= ""
    #print(patent_abstract)
    for _id in patent_abstract:
        #print(i)
        if stopwords_bool:
            abst = clean(patent_abstract[_id],stopwords)
        else:   
            abst = clean_tokenize(patent_abstract[_id],stopwords)
        
        #tab_abst[_id] = abst
        sum_vect_embeddings[_id]= np.zeros(dim,float)
        embeddings1 = model.encode(abst, convert_to_tensor=True)
        #embeddings = model.encode(texts)
        #print(embeddings1)
        #sum_vect_embeddings[_id] = embeddings1.detach().numpy()
        sum_vect_embeddings[_id] = embeddings1.cpu().detach().numpy()
        del embeddings1
        #print(embeddings1.detach().numpy())
        

    return(sum_vect_embeddings)

def sum_webpages(product_directory, test):
    tab_vocab = {}
    sum_vect_embeddings_products = {}
    web_pages = load_product_web_pages(product_directory)
    print("Total number of  webpages")
    print(len(web_pages))


    tab_found_symbols = {}
    for page in web_pages:
        print(page)
        symbol = (get_symbol_from_url(page, product_directory)).rstrip()
        
        if symbol in test:
            if symbol not in tab_found_symbols and symbol not in sum_vect_embeddings_products:
                tab_found_symbols[symbol] = symbol
                        
                sum_vect_embeddings_products[symbol]= np.zeros(dim,float)
            
            #print(symbol)
            path_file = page
            f = open(os.path.join(path_file))
            tab_words = {}
            for line in f:
                line = (' '.join(nltk.word_tokenize(line.strip().lower())))
                if len(line.strip()) > 0:

                    #print(line)
                    values = line.split(' ')
                    for w in values:
                        
                        if w not in stopwords and w not in string.punctuation:
                            #print(w)

                            tab_words[w] = w

            # aggregate:
            distinct = {}
            cpt_token = 0
            for w in tab_words:
                tab_vocab[w] = 0
                if w in embedding_index:# and w not in distinct:
                    cpt_token += 1
                    tab_vocab[w] = 1
                    sum_vect_embeddings_products[symbol] = np.add(sum_vect_embeddings_products[symbol],embedding_index[w])
                    distinct[w] = w

            #sum_vect_embeddings_products[symbol] = [x / cpt_token for x in sum_vect_embeddings_products[symbol]]       
                #word = values[0]
                #coefs=np.asarray(values[1:],dtype='float32')
                #embedding_index[word]=coefs
            f.close()
        
        
    return(sum_vect_embeddings_products,tab_vocab)


def sum_webpages_sbert(product_directory, test):

    #dim = 300
    tab_vocab = {}
    sum_vect_embeddings_tmp = {}
    
    #model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    model = SentenceTransformer('all-mpnet-base-v2', device='cpu')
    model.max_seq_length = 512

    tab_desc = {}
    web_pages = load_product_web_pages(product_directory)

    print(len(web_pages))

    tab_found_symbols = {}
    cpt_count = 0
    for page in web_pages:
        

        #print(page)
        symbol = get_symbol_from_url(page, product_directory)
        #print(symbol)
            
        if symbol in test:
            cpt_count +=1
            print(cpt_count)        
            #print(symbol)
            path_file = page
            f = open(os.path.join(path_file))
            tab_words = {}
            concat_desc = ""
            for line in f:
                line = (' '.join(nltk.word_tokenize(line.strip().lower())))
                #line = clean_(line,embedding_index)
                if len(line.strip()) > 0:
                    #print(line)
                        
                    concat_desc = concat_desc + ' ' + line
                
            if  symbol in tab_desc:
                length = len(tab_desc[symbol].split(' ')) + len(concat_desc.split(' '))
                    #print(length)
                tab_desc[symbol] = tab_desc[symbol] + ' ' +  concat_desc    
                
            else:
                tab_desc[symbol] = concat_desc          


            f.close()

        
    for symbol in tab_desc:
        print(symbol)
        sum_vect_embeddings_tmp[symbol]= np.zeros(dim,float)

        desc_all = tab_desc[symbol]
        #print(desc_all)
        if len(desc_all.split(' ')) < 512:
            desc = desc_all
        else:
            desc = prun_page(desc_all)  

        #print("-----------------------")
        #print(desc)    
        embeddings1 = model.encode(desc, convert_to_tensor=True)    
        sum_vect_embeddings_tmp[symbol] = embeddings1.detach().numpy()
        #print(sum_vect_embeddings_tmp[symbol])

        
    return(sum_vect_embeddings_tmp,tab_vocab)

def prun_page(text):
    ch = text.split(' ')
    tab_w = {}
    cpt = 0
    for w in ch:
        if w not in tab_w and cpt < 350:
            if len(w.split())<30:
                if cpt == 0:
                    ch_out = w 
                else:   
                    ch_out = ch_out + ' ' + w
                
                tab_w[w] = w
                cpt += 1
    return(ch_out)

if __name__ == "__main__":



    print('Start')
    args = load_args()
    print(args)

    filter_stopwords = False

    print("From Research and Technology to the Market evaluation using static embeddings...")

    args = load_args()

    path_pretrained_w2v = "/home/andrea/Desktop/work/rcast/conferences/LRECColing2024/data/embeddings/pretrained/en/w2v_GoogleNews-vectors-negative300.bin"
    
    path_technology_w2v = "/home/andrea/Desktop/work/rcast/conferences/naacl_ijcai/data/USPTO/embeddings/w2v_uspto_patents_w5_vec300_min5_sg.txt"
    
    # Load test set
    
    csv.field_size_limit(sys.maxsize)

    file_directory = "../prepare_data/"

    print(args.dev)
    if args.dev == True :
        print("load dev")
        file_name = "dev.csv"
    else:
        print("load test")
        file_name = "test.csv"
    #file_name = "train.csv"
    (test_set, org2patent, org2paper, patent2org, paper2org) = load_train_test_set(file_directory, file_name)

    print(len(test_set))
    print(len(patent2org))
    print(len(paper2org))


    # Load Stop words
    sw_filename = "/home/andrea/Desktop/work/rcast/conferences/LRECColing2024/data/stopwords_en.txt"   
    stopwords = load_stop_words(sw_filename)

    # Variables
    embedding_index = {}

    dim = args.dim # default 300
    minimal_abstract_length = args.min_abstract_length
    
    # load embedding models
    if args.model == "w2v":

        if args.pretrained:
            print("Load pretrained model")
            embedding_index = load_pretrained_w2v(path_pretrained_w2v) 
        else:
            print("Load technology model")
            embedding_index = load_technology_w2v(path_technology_w2v)
        
        print("Size embeddings \t" + str(len(embedding_index)) + " tokens")



    web = 0 
    # Get product description
    #market_path = "../data/org2patent_2015_2023.csv"
    if web == 0:
        market_path = "/home/andrea/Desktop/work/rcast/conferences/LRECColing2024/script/org2patent_All.csv"
        (product_description_test) = load_product_description(market_path,7,test_set)
    #if web ==1:
    #    product_directory = "/home/andrea/Desktop/work/rcast/conferences/released_source_codes_for_conferences/lreccoling2024/tod/data/web/"
        
    #    (product_description_test,tab) = sum_webpages(product_directory,test_set)
    
    #print(len(product_description_test))

    #print((product_description_test))
    min_ = 1000000
    max_ = 0
    count = 0
    sum_ = 0
    for i in product_description_test  :
        abst = product_description_test[i]
        len_ = len(abst.split(' '))
        sum_ += len_
        if len_ > max_:
            max_ = len_
        if len_ < min_:
            min_ = len_     

            

        count +=1
        
    print("Min\t" + str(min_))        
    print("Max\t" + str(max_))
    print("Count\t" + str(count))
    mean = sum_/count
    print("Mean \t" + str(sum_/count))

    delta = 0
    for i in product_description_test:
        abst = product_description_test[i]
        len_ = len(abst.split(' '))
        delta += (abs(len_-mean)**2)


    print("delta **2 \t" + str(delta/count))
    print("delta \t" + str(math.sqrt((delta/count))))            
    #sys.exit()            

    

    #sys.exit()

    patent_path ="../prepare_data/patent_abstract.csv"
    g_patent = load_patents(patent_path,patent2org)
    print(len(g_patent))




    # Get paper abstract
    #paper_path ="/home/andrea/Desktop/work/rcast/conferences/2024/data/patent2paper/paper_abstract.csv"
    paper_path ="../prepare_data/paper_abstract.csv"
    g_paper = load_papers(paper_path,paper2org)
    print(len(g_paper))

    


    (patent2paper, paper2patent) = load_paper_patent_citations(org2patent)




    print(len(patent2paper))

    print(len(paper2patent))    
     




    # Compute: -------------------------------------------------------------------------------------------------------------------------------------------

    if args.model == "w2v":
        dim = 300
        # Patent
        sum_vect_embeddings_patents,tab_abst_patents = sentence_average_embeddings(g_patent, embedding_index, stopwords, dim, args.average, filter_stopwords)
        mean_patent_embeddings_test = sum_embeddings_abst_aggregate(org2patent, sum_vect_embeddings_patents,dim,args.average)


        # Paper
        sum_vect_embeddings_papers,tab_abst_papers = sentence_average_embeddings(g_paper, embedding_index, stopwords, dim, args.average, filter_stopwords)
        mean_papers_embeddings_test = sum_embeddings_abst_aggregate(org2paper, sum_vect_embeddings_papers,dim,args.average)

        # combination (sum)
        mean_combine_embeddings_test = combine_abst_aggregate(org2patent ,mean_patent_embeddings_test, mean_papers_embeddings_test,dim,args.average)

        # Product
        sum_vect_embeddings_products,tab_abst_products = sentence_average_embeddings(product_description_test, embedding_index, stopwords, dim, args.average, filter_stopwords)
        

        print("Score")
        compute_scores(mean_patent_embeddings_test, sum_vect_embeddings_products)

        compute_scores(mean_papers_embeddings_test, sum_vect_embeddings_products)

        compute_scores(mean_combine_embeddings_test, sum_vect_embeddings_products)
        # ---------------------------------------------------------------------------------------------------------------------------------------------------
    if args.model == "bert":
        
        dim = 768
        # Patent
        sum_vect_embeddings_patents,tab_abst_patents = sum_embeddings_BERT(g_patent, stopwords, dim)
    
        mean_patent_embeddings_test = sum_embeddings_abst_aggregate(org2patent, sum_vect_embeddings_patents,dim,args.average)

        # Paper
        #.sum_vect_embeddings_papers,tab_abst_papers = sum_embeddings_BERT(g_paper, stopwords, dim)
        #.mean_papers_embeddings_test = sum_embeddings_abst_aggregate(org2paper, sum_vect_embeddings_papers,dim,args.average)

        # combination (sum)
        #.mean_combine_embeddings_test = combine_abst_aggregate(org2patent ,mean_patent_embeddings_test, mean_papers_embeddings_test,dim,args.average)
        sum_vect_embeddings_products,tab_abst_products = sum_embeddings_BERT(product_description_test, stopwords, dim)

        print("Score")
        compute_scores(mean_patent_embeddings_test, sum_vect_embeddings_products)

        #.compute_scores(mean_papers_embeddings_test, sum_vect_embeddings_products)

        #.compute_scores(mean_combine_embeddings_test, sum_vect_embeddings_products)

    if args.model == "sbert":
        print("ééééééééééééééééééééé")
        #sys.exit()
        #model_type = "mini"
        dim =384
        model_type = "base"
        #dim = 768
        #dim =384
        #dim = 1024
        # Patent
        sum_vect_embeddings_patents,tab_abst_patents = sum_embeddings_SBERT(g_patent,stopwords,filter_stopwords,model_type)
        mean_patent_embeddings_test = sum_embeddings_abst_aggregate(org2patent, sum_vect_embeddings_patents,dim,args.average)

        # Paper
        #.sum_vect_embeddings_papers,tab_abst_papers = sum_embeddings_SBERT(g_paper,stopwords,filter_stopwords,model_type)
        #.mean_papers_embeddings_test = sum_embeddings_abst_aggregate(org2paper, sum_vect_embeddings_papers,dim,args.average)
        web = 0
        # combination (sum)
        #.mean_combine_embeddings_test = combine_abst_aggregate(org2patent ,mean_patent_embeddings_test, mean_papers_embeddings_test,dim,args.average)
        if web == 0:
            sum_vect_embeddings_products,tab_abst_products = sum_embeddings_SBERT(product_description_test,stopwords,filter_stopwords,model_type)
        
        if web ==1:
            product_directory = "/home/andrea/Desktop/work/rcast/conferences/released_source_codes_for_conferences/lreccoling2024/tod/data/web/"
        
            (sum_vect_embeddings_products,tab_abst_products) = sum_webpages_sbert(product_directory,test_set)

        print("Score ")
        compute_scores(mean_patent_embeddings_test, sum_vect_embeddings_products)

        #compute_scores(mean_papers_embeddings_test, sum_vect_embeddings_products)

        #compute_scores(mean_combine_embeddings_test, sum_vect_embeddings_products)

    if args.model == "simcse":
        model_type = "mini"
        dim =384
        model_type = "base"
        dim = 768
        #dim = 1024
        # Patent
        sum_vect_embeddings_patents,tab_abst_patents = sum_embeddings_SimCSE(g_patent,stopwords,filter_stopwords,model_type)
        mean_patent_embeddings_test = sum_embeddings_abst_aggregate(org2patent, sum_vect_embeddings_patents,dim,args.average)

        # Paper
        sum_vect_embeddings_papers,tab_abst_papers = sum_embeddings_SimCSE(g_paper,stopwords,filter_stopwords,model_type)
        mean_papers_embeddings_test = sum_embeddings_abst_aggregate(org2paper, sum_vect_embeddings_papers,dim,args.average)
        web = 0
        # combination (sum)
        mean_combine_embeddings_test = combine_abst_aggregate(org2patent ,mean_patent_embeddings_test, mean_papers_embeddings_test,dim,args.average)
        if web == 0:
            sum_vect_embeddings_products,tab_abst_products = sum_embeddings_SimCSE(product_description_test,stopwords,filter_stopwords,model_type)
        
        if web ==1:
            product_directory = "/home/andrea/Desktop/work/rcast/conferences/released_source_codes_for_conferences/lreccoling2024/tod/data/web/"
        
            (sum_vect_embeddings_products,tab_abst_products) = sum_webpages_sbert(product_directory,test_set)

        print("Score ")
        compute_scores(mean_patent_embeddings_test, sum_vect_embeddings_products)

        compute_scores(mean_papers_embeddings_test, sum_vect_embeddings_products)

        compute_scores(mean_combine_embeddings_test, sum_vect_embeddings_products)    


    if args.model == "specter":
        model_type = "mini"
        dim =384
        model_type = "base"
        dim = 768
        #dim = 1024
        # Patent
        sum_vect_embeddings_patents,tab_abst_patents = sum_embeddings_SPECTER(g_patent,stopwords,filter_stopwords,model_type)
        mean_patent_embeddings_test = sum_embeddings_abst_aggregate(org2patent, sum_vect_embeddings_patents,dim,args.average)

        # Paper
        #sum_vect_embeddings_papers,tab_abst_papers = sum_embeddings_SPECTER(g_paper,stopwords,filter_stopwords,model_type)
        #mean_papers_embeddings_test = sum_embeddings_abst_aggregate(org2paper, sum_vect_embeddings_papers,dim,args.average)
        web = 0
        # combination (sum)
        #mean_combine_embeddings_test = combine_abst_aggregate(org2patent ,mean_patent_embeddings_test, mean_papers_embeddings_test,dim,args.average)
        if web == 0:
            sum_vect_embeddings_products,tab_abst_products = sum_embeddings_SPECTER(product_description_test,stopwords,filter_stopwords,model_type)
        
        if web ==1:# 
            product_directory = "/home/andrea/Desktop/work/rcast/conferences/released_source_codes_for_conferences/lreccoling2024/tod/data/web/"
        
            (sum_vect_embeddings_products,tab_abst_products) = sum_webpages_SPECTER(product_directory,test_set)

        print("Score ")
        compute_scores(mean_patent_embeddings_test, sum_vect_embeddings_products)

        #compute_scores(mean_papers_embeddings_test, sum_vect_embeddings_products)

        #compute_scores(mean_combine_embeddings_test, sum_vect_embeddings_products)        

    if args.model == "specter2":
        model_type = "mini"
        dim =384
        model_type = "base"
        dim = 768
        #dim = 1024
        # Patent
        sum_vect_embeddings_patents,tab_abst_patents = sum_embeddings_SPECTER2(g_patent,stopwords,filter_stopwords,model_type)
        mean_patent_embeddings_test = sum_embeddings_abst_aggregate(org2patent, sum_vect_embeddings_patents,dim,args.average)

        # Paper
        #sum_vect_embeddings_papers,tab_abst_papers = sum_embeddings_SPECTER2(g_paper,stopwords,filter_stopwords,model_type)
        #mean_papers_embeddings_test = sum_embeddings_abst_aggregate(org2paper, sum_vect_embeddings_papers,dim,args.average)
        web = 0
        # combination (sum)
        #mean_combine_embeddings_test = combine_abst_aggregate(org2patent ,mean_patent_embeddings_test, mean_papers_embeddings_test,dim,args.average)
        if web == 0:
            sum_vect_embeddings_products,tab_abst_products = sum_embeddings_SPECTER2(product_description_test,stopwords,filter_stopwords,model_type)
        
        if web ==1:# 
            product_directory = "/home/andrea/Desktop/work/rcast/conferences/released_source_codes_for_conferences/lreccoling2024/tod/data/web/"
        
            (sum_vect_embeddings_products,tab_abst_products) = sum_webpages_SPECTER2(product_directory,test_set)

        print("Score ")
        compute_scores(mean_patent_embeddings_test, sum_vect_embeddings_products)

        #compute_scores(mean_papers_embeddings_test, sum_vect_embeddings_products)

        #compute_scores(mean_combine_embeddings_test, sum_vect_embeddings_products)    

        

    if args.model == "instructor":
        model_type = "mini"
        dim =384
        model_type = "base"
        dim = 768
        #dim = 1024
        # Patent
        sum_vect_embeddings_patents,tab_abst_patents = sum_embeddings_Instructor(g_patent,stopwords,filter_stopwords,model_type, "Patent abstract")
        mean_patent_embeddings_test = sum_embeddings_abst_aggregate(org2patent, sum_vect_embeddings_patents,dim,args.average)

        # Paper
        sum_vect_embeddings_papers,tab_abst_papers = sum_embeddings_Instructor(g_paper,stopwords,filter_stopwords,model_type, "Paper abstract")
        mean_papers_embeddings_test = sum_embeddings_abst_aggregate(org2paper, sum_vect_embeddings_papers,dim,args.average)
        web = 0
        # combination (sum)
        mean_combine_embeddings_test = combine_abst_aggregate(org2patent ,mean_patent_embeddings_test, mean_papers_embeddings_test,dim,args.average)
        if web == 0:
            sum_vect_embeddings_products,tab_abst_products = sum_embeddings_Instructor(product_description_test,stopwords,filter_stopwords,model_type, "Product description")
        
        if web ==1:# 
            product_directory = "/home/andrea/Desktop/work/rcast/conferences/released_source_codes_for_conferences/lreccoling2024/tod/data/web/"
        
            (sum_vect_embeddings_products,tab_abst_products) = sum_webpages_Instructor(product_directory,test_set)

        print("Score ")
        compute_scores(mean_patent_embeddings_test, sum_vect_embeddings_products)

        compute_scores(mean_papers_embeddings_test, sum_vect_embeddings_products)

        compute_scores(mean_combine_embeddings_test, sum_vect_embeddings_products)    

    if args.model == "e5":
        
        dim = 768
        #dim = 1024
        # Patent
        sum_vect_embeddings_patents,tab_abst_patents = mistral_e5(g_patent,stopwords,filter_stopwords)
        mean_patent_embeddings_test = sum_embeddings_abst_aggregate(org2patent, sum_vect_embeddings_patents,dim,args.average)

        # Paper
        sum_vect_embeddings_papers,tab_abst_papers = mistral_e5(g_paper,stopwords,filter_stopwords)
        mean_papers_embeddings_test = sum_embeddings_abst_aggregate(org2paper, sum_vect_embeddings_papers,dim,args.average)

        # combination (sum)
        mean_combine_embeddings_test = combine_abst_aggregate(org2patent ,mean_patent_embeddings_test, mean_papers_embeddings_test,dim,args.average)
        sum_vect_embeddings_products,tab_abst_products = mistral_e5(product_description_test,stopwords,filter_stopwords)

        print("Score ")
        compute_scores(mean_patent_embeddings_test, sum_vect_embeddings_products)

        compute_scores(mean_papers_embeddings_test, sum_vect_embeddings_products)

        compute_scores(mean_combine_embeddings_test, sum_vect_embeddings_products)

    if args.model == "sgpt":
        
        dim = 768
        #dim = 2048
        model_type= "base"
        #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:21"
        

        # Patent
        sum_vect_embeddings_patents = sum_embeddings_SGPT(g_patent,stopwords,filter_stopwords,model_type)
        mean_patent_embeddings_test = sum_embeddings_abst_aggregate(org2patent, sum_vect_embeddings_patents,dim,args.average)

        # Paper
        #sum_vect_embeddings_papers = sum_embeddings_SGPT(g_paper,stopwords,filter_stopwords,model_type)
        #mean_papers_embeddings_test = sum_embeddings_abst_aggregate(org2paper, sum_vect_embeddings_papers,dim,args.average)

        # combination (sum)
       # mean_combine_embeddings_test = combine_abst_aggregate(org2patent ,mean_patent_embeddings_test, mean_papers_embeddings_test,dim,args.average)
        sum_vect_embeddings_products = sum_embeddings_SGPT(product_description_test,stopwords,filter_stopwords,model_type)

        print("Score ")
        compute_scores(mean_patent_embeddings_test, sum_vect_embeddings_products)

        #compute_scores(mean_papers_embeddings_test, sum_vect_embeddings_products)

        #compute_scores(mean_combine_embeddings_test, sum_vect_embeddings_products)