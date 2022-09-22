import fasttext
from bs4 import BeautifulSoup
import re
import nltk
import urllib.request as urllib2
import numpy as np
from googlesearch import search
import pandas as pd
import math
import os

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import torch
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

from summarizer import Summarizer
import nltk

def clean_text(text):
    cleaned_text = re.sub(r'[^A-Za-z0-9 ]+', '', text)
    return cleaned_text.lower()

def get_results_inst(paragraphs, M1):

    results = []

    for p in paragraphs:
        sentences = [s for s in nltk.sent_tokenize(p)]
        selected = [False for i in range(len(sentences))]
        sentences_cleaned = [clean_text(s) for s in sentences]

        pred = [M1.predict(s)[0][0][len("__label__"):][0] for s in sentences_cleaned]
        chunk_idxs = [(m.start(0), m.end(0)) for m in re.finditer("I*", "".join(pred))]
        
        chunks = []
        for start, end in chunk_idxs:
            if start == end:
                continue
            selected[start:end] = [True for i in range(end - start)]
            chunks.append(sentences[start:end])
        
        chunk_text = [" ".join(c) for c in chunks]
        results.append((p, chunk_text, selected))
        
    return results

def get_summary_inst(paragraphs_web, model):
    
    results_web = get_results_inst(paragraphs_web, model)
    instances = [r[1] for r in results_web]
    instances = [" ".join(i) for i in instances]
    instances = [i for i in instances if len(i) > 0]
    
    return instances

def get_results_begend(paragraphs, M2_beg, M2_end):

    results = []

    for p in paragraphs:
        sentences = [s for s in nltk.sent_tokenize(p)]
        sentences_cleaned = [clean_text(s) for s in sentences]

        selected = [False for i in range(len(sentences))]
        
        beg_pred = [(M2_beg.predict(s)[0][0][len("__label__"):] == 'Beginning_Instance', M2_beg.predict(s)[1][0]) for s in sentences_cleaned]
        end_pred = [(M2_end.predict(s)[0][0][len("__label__"):] == 'Ending_Instance', M2_end.predict(s)[1][0]) for s in sentences_cleaned]

        labels = []

        for i in range(len(beg_pred)):
            beg_lab, beg_score = beg_pred[i]
            end_lab, end_score = end_pred[i]
            
            if max(beg_score, end_score) < 0.8:
                labels.append("N")
                continue
            
            if beg_lab and end_lab:
                labels.append("B" if beg_score >= end_score else "E")
            elif beg_lab:
                labels.append("B")
            elif end_lab:
                labels.append("E")
            else:
                labels.append("N")
                
        chunks = []

        for i in range(len(labels)):
            if labels[i] == "B":
                new_chunk = [sentences[i]]
                selected[i] = True
                if i != len(labels) - 1 and labels[i + 1] == "E":
                    new_chunk.append(sentences[i + 1])
                    selected[i + 1] = True
                chunks.append(new_chunk)

        chunk_text = [" ".join(c) for c in chunks]

        results.append((p, chunk_text, selected))

    return results

def get_summary_begend(paragraphs_web, model1, model2):

    results_web = get_results_begend(paragraphs_web, model1, model2)
    instances = [r[1] for r in results_web]
    instances = [" ".join(i) for i in instances]
    instances = [i for i in instances if len(i) > 0]
    
    return instances

def get_results_multi(paragraphs, M3):

    results = []

    for p in paragraphs:
        sentences = [s for s in nltk.sent_tokenize(p)]
        selected = [False for i in range(len(sentences))]
        sentences_cleaned = [clean_text(s) for s in sentences]

        pred = [M3.predict(s)[0][0][len("__label__"):][0] for s in sentences_cleaned]
        chunk_idxs = [(m.start(0), m.end(0)) for m in re.finditer("B*M*E*", "".join(pred))]
        
        chunks = []
        for start, end in chunk_idxs:
            if start == end:
                continue
            selected[start:end] = [True for i in range(end - start)]
            chunks.append(sentences[start:end])
        
        chunk_text = [" ".join(c) for c in chunks]
        results.append((p, chunk_text, selected))
        
    return results

def get_summary_multi(paragraphs_web, model):
        
    results_web = get_results_multi(paragraphs_web, model)
    instances = [r[1] for r in results_web]
    instances = [" ".join(i) for i in instances]
    instances = [i for i in instances if len(i) > 0]
    
    return instances

def get_urls(topic, section):
    query = f"{topic} {section}"
    urls = [url for url in search(query, num_results = 15, lang = "en")]  
    urls = [url for url in urls if ".org" not in url and ".edu" not in url and "wikipedia" not in url]
    return urls[:min(5, len(urls))]

def create_summary_data(topic, section, paragraphs_web):

    path = f"../Instance Classification/models/{topic}/{section}"

    beg_model = fasttext.load_model(f"{path}/beg_model.bin")
    end_model = fasttext.load_model(f"{path}/end_model.bin")

    inst_model = fasttext.load_model(f"{path}/inst_model.bin")

    multi_model = fasttext.load_model(f"{path}/multi_model.bin")

    s1 = get_summary_begend(paragraphs_web, beg_model, end_model)
    s2 = get_summary_inst(paragraphs_web, inst_model)
    s3 = get_summary_multi(paragraphs_web, multi_model)

    return s1, s2, s3

def get_urls(topic, section):
    query = f"{topic} {section}"
    urls = [url for url in search(query, num_results = 15, lang = "en")]  
    urls = [url for url in urls if ".org" not in url and ".edu" not in url and "wikipedia" not in url]
    
    ret = []
    itr = 0
    
    while len(ret) != 5:
        
        try:
            hdr = {'User-Agent': 'Mozilla/5.0'}
            req = urllib2.Request(urls[itr], headers = hdr)
            page = urllib2.urlopen(req, timeout = 10)
            soup = BeautifulSoup(page, "html.parser")

            paragraphs_web = soup.findAll("p")
            paragraphs_web = [p.text for p in paragraphs_web]
            
            ret.append(urls[itr])
        except:
            a = 1
            
        itr += 1
        
    return ret

def clean_lemmatize(text_, stopwords, lemmatizer):

    ret = ""
    cleaned_text = clean_text(text_)
    
    for word in nltk.word_tokenize(cleaned_text):
        if word in stopwords:
            continue
        ret += f"{lemmatizer.lemmatize(word)} "    
    return ret[:-1]

def gen_summaries(topic, section):

    urls = get_urls(topic, section)
    url = urls[0]

    hdr = {'User-Agent': 'Mozilla/5.0'}
    req = urllib2.Request(url, headers = hdr)
    page = urllib2.urlopen(req, timeout = 10)
    soup = BeautifulSoup(page, "html.parser")

    paragraphs_web = soup.findAll("p")
    paragraphs_web = [p.text for p in paragraphs_web]


    s1, s2, s3 = create_summary_data(topic, section, paragraphs_web)
    fasttext_summary = "\n\n".join(s1)

    num_sentences = 0
    length = 0
    for inst in s1:
        length += len(inst)
        sentences = nltk.sent_tokenize(inst)
        num_sentences += len(sentences)
                
    #print(num_sentences, length)

    model_bert = Summarizer()
    summary = model_bert("\n".join(paragraphs_web), num_sentences = num_sentences)

    sent = nltk.sent_tokenize(summary)
    bert_summary = "\n\n".join(sent)

    stopwords = nltk.corpus.stopwords.words('english')
    lemmatizer = nltk.stem.WordNetLemmatizer()

    all_text = []

    for u in urls:
        hdr = {'User-Agent': 'Mozilla/5.0'}
        req = urllib2.Request(u, headers = hdr)
        page = urllib2.urlopen(req, timeout = 10)
        soup = BeautifulSoup(page, "html.parser")

        paragraphs_web = soup.findAll("p")
        paragraphs_web = [p.text for p in paragraphs_web]
        all_text.extend(paragraphs_web)
        
    all_text_clean = [clean_lemmatize(t, stopwords, lemmatizer) for t in all_text]

    tf_idf = TfidfVectorizer()
    all_text_tfidf = tf_idf.fit_transform(all_text_clean)
    all_text_tfidf = tf_idf.transform(all_text_clean)  

    topic_tfidf = tf_idf.transform([clean_lemmatize(f"{topic} {section}", stopwords, lemmatizer)])

    data = [(all_text[i], all_text_tfidf[i], len(nltk.sent_tokenize(all_text[i]))) for i in range(len(all_text))]
    data.sort(key = lambda item: cosine_similarity(item[1], topic_tfidf)[0][0], reverse = True) 

    total_len = 0
    added = []
    itr = 0


    while total_len < num_sentences:
        curr_p, curr_vec, curr_num_sent = data[itr]
        
        can_be_added = True
        total_sim = 0
        for p, vec in added:
            if cosine_similarity(vec, curr_vec)[0][0] > 0.5:
                can_be_added = False
                break
                
        if can_be_added:
            added.append((curr_p, curr_vec))
            total_len += len(nltk.sent_tokenize(curr_p))
            
        itr += 1
        
    p_out = [a[0] for a in added]

    paragraph_summary = "\n\n".join(p_out)

    path1 = f"summaries/{topic}"
    path2 = f"summaries/{topic}/{section}"

    if not os.path.exists(path1):
        os.makedirs(path1)
    if not os.path.exists(path2):
        os.makedirs(path2)

    fasttext_file = open(f"{path2}/fasttext.txt", "w+", encoding = "utf-8")
    bert_file = open(f"{path2}/bert.txt", "w+", encoding = "utf-8")
    paragraph_file = open(f"{path2}/paragraph.txt", "w+", encoding = "utf-8")

    fasttext_file.write(fasttext_summary)
    bert_file.write(bert_summary)
    paragraph_file.write(paragraph_summary)

    fasttext_file.close()
    bert_file.close()
    paragraph_file.close()


wiki_gen_section_data = pd.read_csv("ml_sections.csv", header = None)
for idx, row in wiki_gen_section_data.iterrows():
    topic, section, true_section = row

    print(f"Beginning ({topic}, {section})")

    gen_summaries(topic, section)