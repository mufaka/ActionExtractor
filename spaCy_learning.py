import spacy 
from spacy import displacy

# https://github.com/explosion/spaCy

# sample what spaCy can do for learning purposes. 
# load a model
'''
https://spacy.io/models for 'built in' models
https://spacy.io/universe/category/models for available 3rd party models.

models need to be loaded into the environment before being used
eg:
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_lg
'''
nlp = spacy.load("en_core_web_lg")

text = "As with dogs from other working breeds, the Australian Cattle Dog is energetic and intelligent with an independent streak. It responds well to structured training, particularly if it is interesting and challenging. It was originally bred to herd by biting, and is known to nip running children. It forms a strong attachment to its owners, and can be protective of them and their possessions. It is easy to groom and maintain, requiring little more than brushing during the shedding period. The most common health problems are deafness and progressive blindness (both hereditary conditions) and accidental injury; otherwise, it is a robust breed with a lifespan of 12 to 16 years."

doc = nlp(text)

# iterate the sentences in the text
for number, sent in enumerate(doc.sents):
    print(number, sent)
    for token in sent:
        print(f'{token.i}\t{token}\t{token.pos_}\t{token.tag_}\t{token.morph.to_dict()}')
        print(f'\t---> {token.head.i}\t{token.dep_}\t')
    
# get any named entities in doc
for ent in doc.ents:
    print(f'{ent.text}\t{ent.label_}\t{ent.start}\t{ent.end}')

'''
the Australian Cattle Dog ORG
12 to 16 years DATE
'''


# can't imagine this working while running in VS Code terminal attached to WSL 2 ...
# displacy.render(doc, style='dep', options={'compact': True})

# look at the pipeline
#check the pipeline
print("Pipeline:", nlp.pipe_names)

# Where is the pipeline being loaded from?
# NOTE: on WSL you can go to the \\wsl$\<distro name> and then navigate to the following path to view files (but don't edit!)
print(nlp.path)


