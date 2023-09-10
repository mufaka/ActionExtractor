#pip install 'spacy[transformers]'
#python -m spacy download en_core_web_trf
#python -m spacy download en_core_web_sm
#python -m spacy download en_core_web_lg
import spacy 

def load_trf_nlp():
    return spacy.load("en_core_web_trf")

def load_sm_nlp():
    return spacy.load("en_core_web_sm")

def load_lg_nlp():
    return spacy.load("en_core_web_lg")

def load_doc(nlp, text):
    return nlp(text)

def show_morph(doc):    
    for token in doc:
        print(f'{token.text}\t\t{token.pos_}\t{token.tag_}\t{token.morph.to_dict()}')

nlp = load_trf_nlp()
#nlp = load_sm_nlp()
#nlp = load_lg_nlp()

#check the pipeline
print("Pipeline:", nlp.pipe_names)

show_morph(load_doc(nlp, "I don't watch the news, I read the paper instead"))

# https://spacy.io/usage/linguistic-features#morphology

'''
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
