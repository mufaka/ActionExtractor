#pip install 'spacy[transformers]'
#python -m spacy download en_core_web_trf
#python -m spacy download en_core_web_sm
#python -m spacy download en_core_web_lg
import spacy 
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

def load_doc(nlp, text):
    return nlp(text)

def show_morph(doc):    
    for token in doc:
        print(f'{token.text}\t\t\t{token.pos_}\t{token.tag_}\t{token.dep_}\t{token.shape_}\t{token.is_alpha}\t{token.is_stop}\t{token.morph.to_dict()}')

nlp = load_trf_nlp() 
#nlp = load_sm_nlp()
#nlp = load_lg_nlp() 

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
doc = load_doc(nlp, "Add a modifier of either RT or LT to procedure 30801 and then resubmit the claim.")

show_morph(doc)

# pattern matching?
pattern = [{'POS': 'VERB', 'OP': '?'},{'POS': 'ADV', 'OP': '*'},{'OP': '*'},{'POS': 'VERB', 'OP': '+'}]

matcher = Matcher(nlp.vocab)
matcher.add("verb-phrases", [pattern])
matches = matcher(doc)
spans = [doc[start:end] for _, start, end in matches]
print(pattern)
print(spans)


'''
https://spacy.io/usage/linguistic-features#morphology

I               PRON    PRP     {'Case': 'Nom', 'Number': 'Sing', 'Person': '1', 'PronType': 'Prs'}
do              AUX     VBP     {'Mood': 'Ind', 'Tense': 'Pres', 'VerbForm': 'Fin'}
n't             PART    RB      {'Polarity': 'Neg'}
watch           VERB    VB      {'VerbForm': 'Inf'}
the             DET     DT      {'Definite': 'Def', 'PronType': 'Art'}
news            NOUN    NN      {'Number': 'Sing'}
,               PUNCT   ,       {'PunctType': 'Comm'}
I               PRON    PRP     {'Case': 'Nom', 'Number': 'Sing', 'Person': '1', 'PronType': 'Prs'}
read            VERB    VBP     {'Tense': 'Pres', 'VerbForm': 'Fin'} <--- doesn't include Mood as shown in documentation link above.
the             DET     DT      {'Definite': 'Def', 'PronType': 'Art'}
paper           NOUN    NN      {'Number': 'Sing'}
instead         ADV     RB      {}'''


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
