import spacy 
from spacy import displacy
from spacy.pipeline.morphologizer import DEFAULT_MORPH_MODEL

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
nlp = spacy.load("en_core_web_lg", disable=["lemmatizer", "ner"])
# print(nlp) --> spacy.lang.en.English object .. https://spacy.io/api/language .. 
# print(spacy.info("en_core_web_lg")) --> 'description': 'English pipeline optimized for CPU. Components: tok2vec, tagger, parser, senter, ner, attribute_ruler, lemmatizer.',

# https://spacy.io/api/morphologizer
# the model doesn't define a morphologizer so the tagger does it, but not very well? How do we configure?
config = {"model": DEFAULT_MORPH_MODEL}
#nlp.add_pipe("morphologizer", config=config) #<--- causes ValueError: [E109] Component 'morphologizer' could not be run. Did you forget to call `initialize()`? So initialize on what exactly?
#nlp.initialize() #<-- causes [E955] Can't find table(s) lexeme_norm for language 'en' in spacy-lookups-data. Make sure you have the package installed or provide your own lookup tables if no default lookups are available for your language.

# This https://github.com/explosion/spaCy/issues/7382#issuecomment-795076457 seems to indicate morphologizer shouldn't be used, but change the order of the pipeline?
#   But, there are two issues combined there and it's not clear if the fix is for the first issue or the second....
# This API is all over the place with breaking changes that makes it very difficult to find answers. 

print("Pipeline:", nlp.pipe_names)
# Pipeline: ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']

#quit()

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
print(nlp.pipeline)

'''
Pipeline: ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']
[
    ('tok2vec', <spacy.pipeline.tok2vec.Tok2Vec object at 0x7f9ee8db4bf0>), 
    ('tagger', <spacy.pipeline.tagger.Tagger object at 0x7f9ee8db4e30>), 
    ('parser', <spacy.pipeline.dep_parser.DependencyParser object at 0x7f9ee8d8d5b0>), 
    ('attribute_ruler', <spacy.pipeline.attributeruler.AttributeRuler object at 0x7f9ee7f71210>), 
    ('lemmatizer', <spacy.lang.en.lemmatizer.EnglishLemmatizer object at 0x7f9eea001490>), 
    ('ner', <spacy.pipeline.ner.EntityRecognizer object at 0x7f9ee8d8d770>)
]
'''

# Where is the model being loaded from?
# NOTE: on WSL you can go to the \\wsl$\<distro name> and then navigate to the following path to view files (but don't edit!)
print(nlp.path)
# /home/bnickel/miniconda3/lib/python3.11/site-packages/en_core_web_lg/en_core_web_lg-3.6.0


# more on morph
# https://universaldependencies.org/u/feat/index.html for morph features standards
# https://en.wikipedia.org/wiki/Imperative_mood
imperative_sentences = [
    "Come to the party tomorrow! ",
    "Eat the apple if you want.",
    "Have a nice trip!",
    "I have to ask you to stop.",
    "It would be great if you made us a drink.",
    "Go to your cubicle!",
    "Hit the ball."
]

# what is the vocab?
print(nlp.vocab)

# what is the Morphology being used?
print(nlp.vocab.morphology)

# do we / can we add features?
print(nlp.vocab.morphology.add("Mood=Imp"))

# missing something; not seeing full morph with mood ... grrr
def show_morph(doc):    
    for token in doc:
        print(f'{token.text}\t\t\t{token.pos_}\t{token.tag_}\t{token.dep_}\t{token.shape_}\t{token.is_alpha}\t{token.is_stop}\t{token.morph.to_dict()}')



imperative_docs = list(nlp.pipe(imperative_sentences))

for imperative_doc in imperative_docs:
    show_morph(imperative_doc)

'''
Come                    VERB    VB      ROOT    Xxxx    True    False   {'VerbForm': 'Inf'}
to                      ADP     IN      prep    xx      True    True    {}
the                     DET     DT      det     xxx     True    True    {'Definite': 'Def', 'PronType': 'Art'}
party                   NOUN    NN      pobj    xxxx    True    False   {'Number': 'Sing'}
tomorrow                        NOUN    NN      npadvmod        xxxx    True    False   {'Number': 'Sing'}
!                       PUNCT   .       punct   !       False   False   {'PunctType': 'Peri'}
'''
'''
Eat                     VERB    VB      ROOT    Xxx     True    False   {'VerbForm': 'Inf'}
the                     DET     DT      det     xxx     True    True    {'Definite': 'Def', 'PronType': 'Art'}
apple                   NOUN    NN      dobj    xxxx    True    False   {'Number': 'Sing'}
if                      SCONJ   IN      mark    xx      True    True    {}
you                     PRON    PRP     nsubj   xxx     True    True    {'Case': 'Nom', 'Person': '2', 'PronType': 'Prs'}
want                    VERB    VBP     advcl   xxxx    True    False   {'Tense': 'Pres', 'VerbForm': 'Fin'}
.                       PUNCT   .       punct   .       False   False   {'PunctType': 'Peri'}
'''
'''
Have                    VERB    VB      ROOT    Xxxx    True    True    {'VerbForm': 'Inf'}
a                       DET     DT      det     x       True    True    {'Definite': 'Ind', 'PronType': 'Art'}
nice                    ADJ     JJ      amod    xxxx    True    False   {'Degree': 'Pos'}
trip                    NOUN    NN      dobj    xxxx    True    False   {'Number': 'Sing'}
!                       PUNCT   .       punct   !       False   False   {'PunctType': 'Peri'}
'''
'''
I                       PRON    PRP     nsubj   X       True    True    {'Case': 'Nom', 'Number': 'Sing', 'Person': '1', 'PronType': 'Prs'}
have                    VERB    VBP     ROOT    xxxx    True    True    {'Mood': 'Ind', 'Tense': 'Pres', 'VerbForm': 'Fin'}
to                      PART    TO      aux     xx      True    True    {}
ask                     VERB    VB      xcomp   xxx     True    False   {'VerbForm': 'Inf'}
you                     PRON    PRP     dobj    xxx     True    True    {'Case': 'Acc', 'Person': '2', 'PronType': 'Prs'}
to                      PART    TO      aux     xx      True    True    {}
stop                    VERB    VB      xcomp   xxxx    True    False   {'VerbForm': 'Inf'}
.                       PUNCT   .       punct   .       False   False   {'PunctType': 'Peri'}
'''
'''
It                      PRON    PRP     nsubj   Xx      True    True    {'Case': 'Nom', 'Gender': 'Neut', 'Number': 'Sing', 'Person': '3', 'PronType': 'Prs'}
would                   AUX     MD      aux     xxxx    True    True    {'VerbForm': 'Fin'}
be                      AUX     VB      ROOT    xx      True    True    {'VerbForm': 'Inf'}
great                   ADJ     JJ      acomp   xxxx    True    False   {'Degree': 'Pos'}
if                      SCONJ   IN      mark    xx      True    True    {}
you                     PRON    PRP     nsubj   xxx     True    True    {'Case': 'Nom', 'Person': '2', 'PronType': 'Prs'}
made                    VERB    VBD     advcl   xxxx    True    True    {'Tense': 'Past', 'VerbForm': 'Fin'}
us                      PRON    PRP     nsubj   xx      True    True    {'Case': 'Acc', 'Number': 'Plur', 'Person': '1', 'PronType': 'Prs'}
a                       DET     DT      det     x       True    True    {'Definite': 'Ind', 'PronType': 'Art'}
drink                   NOUN    NN      ccomp   xxxx    True    False   {'Number': 'Sing'}
.                       PUNCT   .       punct   .       False   False   {'PunctType': 'Peri'}
'''
'''
Go                      VERB    VB      ROOT    Xx      True    True    {'VerbForm': 'Inf'}
to                      ADP     IN      prep    xx      True    True    {}
your                    PRON    PRP$    poss    xxxx    True    True    {'Person': '2', 'Poss': 'Yes', 'PronType': 'Prs'}
cubicle                 NOUN    NN      pobj    xxxx    True    False   {'Number': 'Sing'}
!                       PUNCT   .       punct   !       False   False   {'PunctType': 'Peri'}
'''
'''
Hit                     VERB    VB      ROOT    Xxx     True    False   {'VerbForm': 'Inf'}
the                     DET     DT      det     xxx     True    True    {'Definite': 'Def', 'PronType': 'Art'}
ball                    NOUN    NN      dobj    xxxx    True    False   {'Number': 'Sing'}
.                       PUNCT   .       punct   .       False   False   {'PunctType': 'Peri'}
'''
