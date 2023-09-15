import spacy 
import stanza 
import spacy_stanza
from spacy.matcher import Matcher
import pickle

class SpacyAutoMatcher:
    def __init__(self, model_name, key_pattern_file):
        self.model_name = model_name
        
        # stanza translates to stanza.download("en")
        if (model_name == "stanza"):
            self.nlp = self.load_stanza_nlp()
        else :
            self.nlp = spacy.load(model_name)
        
        if key_pattern_file and key_pattern_file != "":
            with open(key_pattern_file, 'rb') as f:
                self.pattern_keys = set(pickle.load(f))

    def load_stanza_nlp():
        stanza.download("en")
        nlp_stanza = spacy_stanza.load_pipeline("en")
        return nlp_stanza

    # find phrases where a noun has an imperative mood verb ancestor
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


    def build_pattern_keys(self, sentences):
        keys = []
        for sentence in sentences:
            doc = self.nlp(sentence)
            key = self.get_doc_token_key(doc)
            if (key != ""):
                if key not in keys:
                    keys.append(key)
        return keys

    def get_doc_token_key(self, doc):        
        pattern = []
        patternKey = []

        # verbs and nouns use tag_, rest use pos_, OP is always + for now
        for token in doc:
            tokenDict = {}
            op = '+'
            if token.pos_ == "NOUN" or token.pos_ == "VERB":
                tokenDict['TAG'] = token.tag_
                patternKey.append(f'TAG-{token.tag_}')
            else: 
                tokenDict['POS'] = token.pos_ 
                patternKey.append(f'POS-{token.pos_}')
                op = '{1}'
            
            tokenDict['OP'] = op
            pattern.append(tokenDict)

        return ':'.join(patternKey) if len(pattern) > 2 else ""

    def get_doc_token_keys(self, doc):        
        pattern = []
        patternKey = []

        # verbs and nouns use tag_, rest use pos_, OP is always + for now
        for token in doc:
            tokenDict = {}
            op = '+'
            if token.pos_ == "NOUN" or token.pos_ == "VERB":
                tokenDict['TAG'] = token.tag_
                patternKey.append(f'TAG-{token.tag_}')
            else: 
                tokenDict['POS'] = token.pos_ 
                patternKey.append(f'POS-{token.pos_}')
                op = '{1}'
            
            tokenDict['OP'] = op
            pattern.append(tokenDict)

        return ':'.join(patternKey) if len(pattern) > 2 else ""

    def get_matching_phrases_by_pattern(self, patterns, text):
        doc = self.nlp(text)        
        matcher = Matcher(self.nlp.vocab)
        matcher.add("verb-phrases", patterns)
        matches = matcher(doc)
        spans = [doc[start:end] for _, start, end in matches]
        return list(spans)

    def get_matching_phrases_by_key(self, text):
        matches = []
        doc = self.nlp(text)
        for sent in doc.sents:
            sent_doc = sent.as_doc()
            key = self.get_doc_token_key(sent_doc)
            if key in self.pattern_keys:
                matches.append(sent.text)
        return matches

    def get_matching_sentence_by_key(self, sentences):
        matches = []
        sentence_docs = list(self.nlp.pipe(sentences))
        for sent_doc in sentence_docs:
            key = self.get_doc_token_key(sent_doc)
            if key in self.pattern_keys:
                matches.append(sent_doc.text)
        return matches
