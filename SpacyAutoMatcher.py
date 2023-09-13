import spacy 
from spacy.matcher import Matcher

class SpacyAutoMatcher:
    def __init__(self, model_name):
        self.model_name = model_name
        self.nlp = spacy.load(model_name)
    
    def build_patterns(self, sentences):
        patterns = []
        patternKeys = {}
        for sentence in sentences:
            doc = self.nlp(sentence)
            key, pattern = self.get_token_pattern(doc)
            if key not in patternKeys:
                patternKeys[key] = ''
                patterns.append(pattern)
        return patterns                

    def get_token_pattern(self, doc):
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
                op = '{1,2}'
            
            tokenDict['OP'] = op
            pattern.append(tokenDict)
        
        return ':'.join(patternKey), pattern 

    def get_matching_phrases(self, exemplarySentences, text):
        patterns = self. build_patterns(exemplarySentences)
        return self.get_matching_phrases_by_pattern(patterns, text)

    def get_matching_phrases_by_pattern(self, patterns, text):
        doc = self.nlp(text)        
        matcher = Matcher(self.nlp.vocab)
        matcher.add("verb-phrases", patterns)
        matches = matcher(doc)
        spans = [doc[start:end] for _, start, end in matches]
        return list(spans)
