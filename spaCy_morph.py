#pip install 'spacy[transformers]'
#python -m spacy download en_core_web_trf
#python -m spacy download en_core_web_sm
#python -m spacy download en_core_web_lg
import spacy 
import stanza 
import spacy_stanza
from spacy.matcher import Matcher

def load_trf_nlp():
    # Pipeline: ['transformer', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner'] 
    return spacy.load("en_core_web_trf")

def load_sm_nlp():
    return spacy.load("en_core_web_sm")

def load_lg_nlp():
    # Pipeline: ['tok2vec', 'tagger', 'parser', 'attribute_ruler', 'lemmatizer', 'ner']
    # need to add entity_ruler before ner before adding morph? https://github.com/explosion/spaCy/issues/7382
    nlp_lg = spacy.load("en_core_web_lg")
    nlp_lg.add_pipe("entity_ruler", before="ner")
    return nlp_lg

def load_stanza_nlp():
    stanza.download("en")
    nlp_stanza = spacy_stanza.load_pipeline("en")
    return nlp_stanza

def load_doc(nlp, text):
    return nlp(text)

def show_morph(doc):    
    for token in doc:
        print(f'{token.text}\t\t\t{token.pos_}\t{token.tag_}\t{token.dep_}\t{token.shape_}\t{token.is_alpha}\t{token.is_stop}\t{token.morph.to_dict()}')

#nlp = load_trf_nlp() 
#nlp = load_sm_nlp()
#nlp = load_lg_nlp() 
nlp = load_stanza_nlp()

'''
The following causes a ValueError: [E109] Component 'morphologizer' could not be run. Did you forget to call `initialize()`?
Dead in the water if we can't get the morph to work correctly ...
'''
#nlp.add_pipe("morphologizer") 

#check the pipeline
print("Pipeline:", nlp.pipe_names)

# NOTE: on WSL you can go to the \\wsl$\<distro name> and then navigate to the following path to view files (but don't edit!)
print(nlp.path)

#doc = load_doc(nlp, "I don't watch the news, I read the paper instead") 
#doc = load_doc(nlp, "Write an essay on morphological features of spaCy.") # Write  VERB    VB      ROOT    Xxxxx   True    False   {'VerbForm': 'Inf'} <-- meh....
#doc = load_doc(nlp, "The author was staring pensively as she wrote")
doc = load_doc(nlp, "You should drive to the store and buy some groceries and then drive back home")
#doc = load_doc(nlp, "What a great day today!") # nominal sentence; shouldn't be considered...

def get_imperative_phrases(nlp, doc):
    verb_phrases = []
    for token in doc:
        if (token.pos_.lower() == "noun"):
            ancestors = list(token.ancestors)
            children = list(token.children)
            verb_phrase = []
            if len(ancestors) > 0:
                if ancestors[0].pos_.lower() == "verb":
                    verb_phrase.append(token)
                    verb_phrase.append(ancestors[0])
                    if len(children) > 0:
                        for child in children:
                            verb_phrase.append(child)
                    sorted_phrase = sorted(verb_phrase, key=lambda x: x.i)
                    verb_phrases.append(sorted_phrase)

    imperative_phrases = []

    # is the phrase imperative?
    for verb_phrase in verb_phrases:
        phrase = " ".join([x.text for x in verb_phrase])
        phrase_doc = nlp(phrase)
        if "Mood=Imp" in phrase_doc[0].morph:
            imperative_phrases.append(phrase)
    
    return imperative_phrases

actions = get_imperative_phrases(nlp, doc)
print(*actions, sep = "\n")

quit()

def find_root_of_sentence(doc):
    root_token = None
    for token in doc:
        if (token.dep_ == "ROOT" or token.dep_ == "root"):
            root_token = token
    return root_token

root = find_root_of_sentence(doc)
print(f'{root.i}\t{root.text}\t\t{root.dep_}')

for token in doc:
    ancestors = [f'{t.text}-{t.i}' for t in token.ancestors]
    children = [f'{t.text}-{t.i}' for t in token.children]
    print(token.text, "\t", token.i, "\t", 
          token.pos_, "\t", token.dep_, "\t", 
          ancestors, "\t", children)

show_morph(doc)

'''
https://spacy.io/usage/linguistic-features#morphology

Drive                   VERB    VB      root    Xxxxx   True    False   {'Mood': 'Imp', 'VerbForm': 'Fin'}
to                      ADP     IN      case    xx      True    True    {}
the                     DET     DT      det     xxx     True    True    {'Definite': 'Def', 'PronType': 'Art'}
store                   NOUN    NN      obl     xxxx    True    False   {'Number': 'Sing'}
and                     CCONJ   CC      cc      xxx     True    True    {}
buy                     VERB    VB      conj    xxx     True    False   {'Mood': 'Imp', 'VerbForm': 'Fin'}
some                    DET     DT      det     xxxx    True    True    {}
groceries                       NOUN    NNS     obj     xxxx    True    False   {'Number': 'Plur'}

pattern is imperative verb to noun

'''

'''
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
'''            

'''
#explain labels
for label in nlp.get_pipe("tagger").labels:
    print(label, " -- ", spacy.explain(label))

$  --  symbol, currency
''  --  closing quotation mark
,  --  punctuation mark, comma
-LRB-  --  left round bracket
-RRB-  --  right round bracket
.  --  punctuation mark, sentence closer
:  --  punctuation mark, colon or ellipsis
ADD  --  email
AFX  --  affix
CC  --  conjunction, coordinating
CD  --  cardinal number
DT  --  determiner
EX  --  existential there
FW  --  foreign word
HYPH  --  punctuation mark, hyphen
IN  --  conjunction, subordinating or preposition
JJ  --  adjective (English), other noun-modifier (Chinese)
JJR  --  adjective, comparative
JJS  --  adjective, superlative
LS  --  list item marker
MD  --  verb, modal auxiliary
NFP  --  superfluous punctuation
NN  --  noun, singular or mass
NNP  --  noun, proper singular
NNPS  --  noun, proper plural
NNS  --  noun, plural
PDT  --  predeterminer
POS  --  possessive ending
PRP  --  pronoun, personal
PRP$  --  pronoun, possessive
RB  --  adverb
RBR  --  adverb, comparative
RBS  --  adverb, superlative
RP  --  adverb, particle
SYM  --  symbol
TO  --  infinitival "to"
UH  --  interjection
VB  --  verb, base form
VBD  --  verb, past tense
VBG  --  verb, gerund or present participle
VBN  --  verb, past participle
VBP  --  verb, non-3rd person singular present
VBZ  --  verb, 3rd person singular present
WDT  --  wh-determiner
WP  --  wh-pronoun, personal
WP$  --  wh-pronoun, possessive
WRB  --  wh-adverb
XX  --  unknown
``  --  opening quotation mark
'''
