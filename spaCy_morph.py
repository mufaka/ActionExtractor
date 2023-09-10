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
        print(f'{token.text} {token.morph}')

#nlp = load_trf_nlp()
nlp = load_sm_nlp()
#nlp = load_lg_nlp()

show_morph(load_doc(nlp, "I don't watch the news, I read the paper"))

# https://spacy.io/usage/linguistic-features#morphology

'''
I           Case=Nom|Number=Sing|Person=1|PronType=Prs
do          Mood=Ind|Tense=Pres|VerbForm=Fin
n't         Polarity=Neg
watch       VerbForm=Inf
the         Definite=Def|PronType=Art
news        Number=Sing
,           PunctType=Comm
I           Case=Nom|Number=Sing|Person=1|PronType=Prs
read        Tense=Past|VerbForm=Fin <--- Doesn't match linked documentation above. Tense should be Pres and missing the all important Mood! 
the         Definite=Def|PronType=Art
paper       Number=Sing
'''


'''
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
            token.shape_, token.is_alpha, token.is_stop)
'''            