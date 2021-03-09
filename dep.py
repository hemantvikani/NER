# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 20:29:46 2021

@author: power
"""



import streamlit as st 
import spacy

#load the trained model from the directory where it is saved 
nlp = spacy.load('E:/entity_reco1/')
        
from urllib.request import urlopen
from bs4 import BeautifulSoup
# Specify url of the web page
source = urlopen('https://en.wikipedia.org/wiki/Walmart').read()
# Make a soup 
soup = BeautifulSoup(source,'lxml')
        
        
print(set([text.parent.name for text in soup.find_all(text=True)]))
        
# Extract the plain text content from paragraphs
text = ''
for paragraph in soup.find_all('p'):
    text += paragraph.text
            
# Import package 
import re
# Clean text
text = re.sub(r'\[.*?\]+', '', text)
        
l = ['\n',"\'s",'\xa0',"\'d","\'t","\'"]
for i in l:
    text = text.replace(i, ' ')


def welcome(): 
    return 'welcome all'

# defining the function which will make the prediction using 
# the data which the user inputs 
def prediction(nlp,text): 

    doc1 = nlp(text)
    print("Entities", [(ent.text, ent.label_) for ent in doc1.ents])
    return doc1
    

# this is the main function in which we define our webpage 
def main(): 
    # giving the webpage a title 
    st.title("Named Entity Recognition") 
    
    # here we define some of the front end elements of the web page like 
    # the font and background color, the padding and the text to be displayed 
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit NER App </h1> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html = True)
    doc = prediction(nlp, text)
    
    st.success( ("Entities", [(ent.text, ent.label_) for ent in doc.ents])) 
    
if __name__=='__main__': 
    main() 
