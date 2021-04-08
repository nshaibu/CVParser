import pandas as pd
import json
from tqdm import tqdm
import spacy
from spacy.tokens import DocBin

nlp = spacy.blank("en") # load a new spacy model
db = DocBin() # create a DocBin object

with open("train.json", "r") as fd:
    TRAIN_DATA = json.loads(fd.read())


for text, annot in tqdm(TRAIN_DATA): # data in previous format
    doc = nlp.make_doc(text) # create doc object from text
    ents = []
    print(text, annot)
    for start, end, label in annot["entities"]: # add character indexes
        span = doc.char_span(start, end, label=label, alignment_mode="contract")
        if span is None:
            print("Skipping entity")
        else:
            ents.append(span)
    try:
        doc.ents = ents # label the text with the ents
    except Exception:
        pass
    db.add(doc)

db.to_disk("./train.spacy") # save the docbin object
