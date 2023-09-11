import spacy 
from spacy import displacy
from spacy.pipeline.morphologizer import DEFAULT_MORPH_MODEL

'''
https://github.com/explosion/spaCy
https://spacy.io/models for 'built in' models
https://spacy.io/universe/category/models for available 3rd party models.

models need to be loaded into the environment before being used
eg:
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_lg
'''

# show information about the pipeline being used
def show_pipeline_info(nlp):
    print("############ PIPELINE INFO ############")
    print(nlp) # eg: --> spacy.lang.en.English object (https://spacy.io/api/language)
    print(nlp.pipe_names)
    print(nlp.pipeline)
    print(nlp.path) # NOTE: on WSL you can go to the \\wsl$\<distro name> and then navigate to the path to view files (but don't edit!)
    print(nlp.vocab) # what is the vocab?
    print(nlp.vocab.morphology) # what is the Morphology being used?
    print()

def show_model_info(model):
    print("############ MODEL INFO ############")
    print(spacy.info("en_core_web_lg")) # eg prop: 'description': 'English pipeline optimized for CPU. Components: tok2vec, tagger, parser, senter, ner, attribute_ruler, lemmatizer.',
    print()

# iterate and print all sentences in a doc
def show_sentences(doc):
    print("############ SENTENCE INFO ############")
    for number, sent in enumerate(doc.sents):
        print(number, sent)
        for token in sent:
            print(f'{token.i}\t{token}\t{token.pos_}\t{token.tag_}\t{token.morph.to_dict()}')
            print(f'\t---> {token.head.i}\t{token.dep_}\t')
        print()
    print()

# iterate and print all named entities in a doc
def show_named_entities(doc):
    print("############ NAMED ENTITIES ############")
    for ent in doc.ents:
        print(f'{ent.text}\t{ent.label_}\t{ent.start}\t{ent.end}')
    print()

# show morph information for a doc
def show_morph(doc):    
    print("############ MORPH ############")
    print(doc)
    for token in doc:
        print(f'{token.text}\t\t\t{token.pos_}\t{token.tag_}\t{token.dep_}\t{token.shape_}\t{token.is_alpha}\t{token.is_stop}\t{token.morph.to_dict()}')
    print()

def show_morph_for_sentences(sentences):
    sentence_docs = list(nlp.pipe(sentences))
    for sentence_doc in sentence_docs:
        show_morph(sentence_doc)

model_name = "en_core_web_trf"
nlp = spacy.load(model_name)

show_model_info(model_name)
show_pipeline_info(nlp)

text = "As with dogs from other working breeds, the Australian Cattle Dog is energetic and intelligent with an independent streak. It responds well to structured training, particularly if it is interesting and challenging. It was originally bred to herd by biting, and is known to nip running children. It forms a strong attachment to its owners, and can be protective of them and their possessions. It is easy to groom and maintain, requiring little more than brushing during the shedding period. The most common health problems are deafness and progressive blindness (both hereditary conditions) and accidental injury; otherwise, it is a robust breed with a lifespan of 12 to 16 years."
doc = nlp(text)

show_morph(doc)

# https://universaldependencies.org/u/feat/index.html for morph features standards
# https://en.wikipedia.org/wiki/Imperative_mood
imperative_sentences = [
    "Drive to the store.",
    "Eat the apple if you want.",
    "Have a nice trip!",
    "I have to ask you to stop.",
    "It would be great if you made us a drink.",
    "Go to your cubicle!",
    "Hit the ball."
]

show_morph_for_sentences(imperative_sentences)

# https://spacy.io/api/morphologizer
# the model doesn't define a morphologizer so the tagger does it, but not very well? How do we configure?
# config = {"model": DEFAULT_MORPH_MODEL}
# nlp.add_pipe("morphologizer", config=config) #<--- causes ValueError: [E109] Component 'morphologizer' could not be run. Did you forget to call `initialize()`? So initialize on what exactly?
# nlp.initialize() #<-- causes [E955] Can't find table(s) lexeme_norm for language 'en' in spacy-lookups-data. Make sure you have the package installed or provide your own lookup tables if no default lookups are available for your language.
# This https://github.com/explosion/spaCy/issues/7382#issuecomment-795076457 seems to indicate morphologizer shouldn't be used, but change the order of the pipeline?
#   But, there are two issues combined there and it's not clear if the fix is for the first issue or the second....
# This API is all over the place with breaking changes that makes it very difficult to find answers. 
