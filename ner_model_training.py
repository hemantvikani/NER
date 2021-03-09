# -*- coding: utf-8 -*-
"""
Created on Mon Mar  8 19:45:26 2021

@author: power
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 15:46:22 2021

@author: power
"""

# Load pre-existing spacy model
import spacy
nlp=spacy.load('en_core_web_sm')

# Getting the pipeline component
ner=nlp.get_pipe("ner")


# training data
TRAIN_DATA = [
              ("Walmart is a leading e-commerce company", {"entities": [(0, 7, "ORG")]}),
              ("I reached Chennai yesterday.", {"entities": [(19, 28, "GPE")]}),
              ("I recently ordered a book from Amazon", {"entities": [(24,32, "ORG")]}),
              ("I was driving a BMW", {"entities": [(16,19, "PRODUCT")]}),
              ("I ordered this from ShopClues", {"entities": [(20,29, "ORG")]}),
              ("Fridge can be ordered in Amazon ", {"entities": [(0,6, "PRODUCT")]}),
              ("I bought a new Washer", {"entities": [(16,22, "PRODUCT")]}),
              ("I bought a old table", {"entities": [(16,21, "PRODUCT")]}),
              ("I bought a fancy dress", {"entities": [(18,23, "PRODUCT")]}),
              ("I rented a camera", {"entities": [(12,18, "PRODUCT")]}),
              ("I rented a tent for our trip", {"entities": [(12,16, "PRODUCT")]}),
              ("I rented a screwdriver from our neighbour", {"entities": [(12,22, "PRODUCT")]}),
              ("I repaired my computer", {"entities": [(15,23, "PRODUCT")]}),
              ("I got my clock fixed", {"entities": [(16,21, "PRODUCT")]}),
              ("I got my truck fixed", {"entities": [(16,21, "PRODUCT")]}),
              ("Flipkart started it's journey from zero", {"entities": [(0,8, "ORG")]}),
              ("I recently ordered from Max", {"entities": [(24,27, "ORG")]}),
              ("Flipkart is recognized as leader in market",{"entities": [(0,8, "ORG")]}),
              ("I recently ordered from Swiggy", {"entities": [(24,29, "ORG")]}),
	("The company was founded by Sam Walton in 1962 and incorporated on October 31, 1969",{"entities": [(41,45, "DATE")]}),
("It also owns and operates Sam  Club retail warehouses",{"entities": [(26,29, "PERSON")]}),
("As of January 31, 2021, Walmart has 11,443 stores and clubs in 27 countries, operating under 56 different names",{"entities": [(24,31, "ORG")]}),
("The company operates under the name Walmart in the United States and Canada, as Walmart de México y Centroamérica in Mexico and Central America, as Asda in the United Kingdom, as the Seiyu Group in Japan, and as Flipkart Wholesale in India",{"entities": [(117,123, "GPE")]}),
("It has wholly owned operations in Argentina, Chile, Canada, and South Africa",{"entities": [(34,43, "GPE")]}),
("Since August 2018, Walmart holds only a minority stake in Walmart Brasil, which was renamed Grupo Big in August 2019, with 20 percent of the company  shares, and private equity firm Advent International holding 80 percent ownership of the company",{"entities": [(6,17, "DATE")]}),
("Walmart is the world  largest company by revenue, with US$548.743 billion, according to the Fortune Global 500 list in 2020",{"entities": [(0,7, "GPE")]}),
("Sam Walton  heirs own over 50 percent of Walmart through both their holding company Walton Enterprises and their individual holdings",{"entities": [(84,102, "ORG")]}),
("Walmart was the largest United States grocery retailer in 2019, and 65 percent of Walmart  US$510.329 billion sales came from U.S",{"entities": [(24,37, "GPE")]}),
("Walmart was listed on the New York Stock Exchange in 1972",{"entities": [(53,57, "DATE")]}),
("By 1988, it was the most profitable retailer in the U.S., and it had become the largest in terms of revenue by October 1989",{"entities": [(52,56, "GPE")]}),
("The company was originally geographically limited to the South and lower Midwest, but it had stores from coast to coast by the early 1990s",{"entities": [(133,138, "DATE")]}),
("Sam  Club opened in New Jersey in November 1989, and the first California outlet opened in Lancaster, in July 1990",{"entities": [(20,30, "GPE")]}),
("A Walmart in York, Pennsylvania, opened in October 1990, the first main store in the Northeast",{"entities": [(19,31, "GPE")]}),
("Walmart  investments outside the U.S",{"entities": [(33,36, "GPE")]}),
("Its operations and subsidiaries in Canada, the United Kingdom, Central America, South America, and China are highly successful, but its ventures failed in Germany and South Korea",{"entities": [(80,93, "GPE")]}),
("On July 2, 1962, Walton opened the first Walmart Discount City store at 719 W",{"entities": [(3,15, "DATE")]}),
("During this year, the first Walmart Supercenter opened in Washington, MO",{"entities": [(58,72, "GPE")]})

              ]


# Adding labels to the `ner`

for _, annotations in TRAIN_DATA:
  for ent in annotations.get("entities"):
    ner.add_label(ent[2])
    
    
# Disable pipeline components you dont need to change
pipe_exceptions = ["ner", "trf_wordpiecer", "trf_tok2vec"]
unaffected_pipes = [pipe for pipe in nlp.pipe_names if pipe not in pipe_exceptions]


# Import requirements
import random
from spacy.util import minibatch, compounding
from pathlib import Path
from spacy.training import Example

# TRAINING THE MODEL

with nlp.disable_pipes(*unaffected_pipes):
    
    optimizer = nlp.resume_training()
   
      # Training for 30 iterations
    for iteration in range(100):

        # shuufling examples  before every iteration
        random.shuffle(TRAIN_DATA)
        losses = {}
  
        # batch up the examples using spaCy's minibatch
        batches = minibatch(TRAIN_DATA, size=compounding(4.0, 32.0, 1.001))
        for batch in batches:
            examples =[]
            for text,annots in batch:
                examples.append(Example.from_dict(nlp.make_doc(text),annots))
                nlp.update(examples,drop=0.5,sgd=optimizer,losses=losses)
        


            print("Losses", losses)
for text, _ in TRAIN_DATA:
    doc = nlp(text)
    print("Entities", [(ent.text, ent.label_) for ent in doc.ents])

#save the model to the particular directory      
output_dir = Path('E:/entity_reco1/')

nlp.to_disk(output_dir)
print("Saved model to", output_dir)




