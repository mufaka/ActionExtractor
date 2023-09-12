import spacy 
from spacy.matcher import Matcher

'''
https://github.com/explosion/spaCy
https://spacy.io/models for 'built in' models
https://spacy.io/universe/category/models for available 3rd party models.

models need to be loaded into the environment before being used
eg:
python -m spacy download en_core_web_trf
python -m spacy download en_core_web_lg
'''
class SpacyAnalyzer:

    def __init__(self, model_name):
        self.model_name = model_name
        self.nlp = spacy.load(model_name)

    def show_pipeline_info(self):
        print("############ PIPELINE INFO ############")
        print(self.nlp) # eg: --> spacy.lang.en.English object (https://spacy.io/api/language)
        print(self.nlp.pipe_names)
        print(self.nlp.pipeline)
        print(self.nlp.path) # NOTE: on WSL you can go to the \\wsl$\<distro name> and then navigate to the path to view files (but don't edit!)
        print(self.nlp.vocab) # what is the vocab?
        print(self.nlp.vocab.morphology) # what is the Morphology being used?
        print()

    def show_model_info(self):
        print("############ MODEL INFO ############")
        print(spacy.info(self.model_name)) # eg prop: 'description': 'English pipeline optimized for CPU. Components: tok2vec, tagger, parser, senter, ner, attribute_ruler, lemmatizer.',
        print()

    def show_sentences(self, doc):
        print("############ SENTENCE INFO ############")
        for number, sent in enumerate(doc.sents):
            print(number, sent)
            for token in sent:
                print(f'{token.i}\t{token}\t{token.pos_}\t{token.tag_}\t{token.morph.to_dict()}')
                print(f'\t---> {token.head.i}\t{token.dep_}\t')
            print()
        print()
    
    def show_named_entities(self, doc):
        print("############ NAMED ENTITIES ############")
        for ent in doc.ents:
            print(f'{ent.text}\t{ent.label_}\t{ent.start}\t{ent.end}')
        print()

    def show_morph(self, doc):    
        print("############ MORPH ############")
        print(doc)
        for token in doc:
            print(f'{token.text}\t\t\t{token.lemma_}\t{token.pos_}\t{token.tag_}\t{token.dep_}\t{token.shape_}\t{token.is_alpha}\t{token.is_stop}\t{token.morph.to_dict()}')
        print()

    def show_morph_for_sentences(self, sentences):
        sentence_docs = list(self.nlp.pipe(sentences))
        for sentence_doc in sentence_docs:
            self.show_morph(sentence_doc)

    def show_verb_phrases(self, doc):
        print("############ VERB PHRASES ############")
        pattern = [
            {'POS': 'VERB', 'OP': '?'},
            {'POS': 'ADV', 'OP': '*'},
            {'OP': '*'},
            {'POS': 'VERB', 'OP': '+'}
        ]
        matcher = Matcher(self.nlp.vocab)
        matcher.add("verb-phrases", [pattern])
        matches = matcher(doc)
        spans = [doc[start:end] for _, start, end in matches]
        print(doc)
        print(spans)

    def show_verb_phrases_for_sentences(self, sentences):
        sentence_docs = list(self.nlp.pipe(sentences))
        for sentence_doc in sentence_docs:
            self.show_verb_phrases(sentence_doc)

    def show_imperative_phrases(self, doc):
        print("############ IMPERATIVE PHRASES ############")
        imperative_phrases = self.get_imperative_phrases(doc)
        for imperative_phrase in imperative_phrases:
            print(imperative_phrase)

    def get_imperative_phrases(self, doc):
        patterns = [
            [
                {'TAG': 'VB', 'OP': '+'},
                {'POS': 'ADP', 'OP': '+'},
                {'POS': 'DET', 'OP': '+'},
                {'TAG': 'NN', 'OP': '+'}
            ],
            [
                {'TAG': 'VB', 'OP': '+'},
                {'POS': 'ADP', 'OP': '+'},
                {'POS': 'DET', 'OP': '+'},
                {'POS': 'ADJ', 'OP': '+'},
                {'TAG': 'PROPN', 'OP': '+'}
            ],
            [
                {'TAG': 'VB', 'OP': '+'},
                {'POS': 'ADP', 'OP': '+'},
                {'POS': 'PRON', 'OP': '+'},
                {'TAG': 'NN', 'OP': '+'}
            ],
            [
                {'TAG': 'VB', 'OP': '+'},
                {'POS': 'ADP', 'OP': '+'},
                {'TAG': 'NN', 'OP': '+'}
            ],
            [
                {'TAG': 'VB', 'OP': '+'},
                {'POS': 'PRON', 'OP': '+'},
                {'POS': 'DET', 'OP': '+'},
                {'TAG': 'NN', 'OP': '+'}
            ],
            [
                {'TAG': 'VB', 'OP': '+'},
                {'POS': 'PRON', 'OP': '+'},
                {'POS': 'ADJ', 'OP': '+'},
                {'TAG': 'NN', 'OP': '+'}
            ],
            [
                {'TAG': 'VB', 'OP': '+'},
                {'POS': 'PRON', 'OP': '+'},
                {'POS': 'DET', 'OP': '+'},
                {'POS': 'ADJ', 'OP': '+'},
                {'TAG': 'NN', 'OP': '+'}
            ],
            [
                {'TAG': 'VB', 'OP': '+'},
                {'POS': 'PRON', 'OP': '+'},
                {'TAG': 'NN', 'OP': '+'}
            ],
            [
                {'TAG': 'VB', 'OP': '+'},
                {'POS': 'DET', 'OP': '+'},
                {'TAG': 'NN', 'OP': '+'}
            ]
        ]
        matcher = Matcher(self.nlp.vocab)
        matcher.add("verb-phrases", patterns)
        matches = matcher(doc)
        spans = [doc[start:end] for _, start, end in matches]
        return list(spans)

    def show_imperative_phrases_for_sentences(self, sentences):
        sentence_docs = list(self.nlp.pipe(sentences))
        for sentence_doc in sentence_docs:
            self.show_imperative_phrases(sentence_doc)

    def get_imperative_phrases_for_sentences(self, sentences):
        imperative_phrases = []
        sentence_docs = list(self.nlp.pipe(sentences))
        for sentence_doc in sentence_docs:
            imperatives = self.get_imperative_phrases(sentence_doc)
            if len(imperatives) > 0:
                imperative_phrases.append(imperatives)
        return imperative_phrases
