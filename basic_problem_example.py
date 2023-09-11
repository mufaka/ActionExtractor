'''
Prerequisites

pip install spacy
python -m spacy download en_core_web_lg
'''
import spacy

nlp = spacy.load("en_core_web_lg")

def show_morph_as_markdown_table(doc):
    print("|Context|Token|Lemma|POS|TAG|MORPH|")
    print("|----|----|----|----|----|----|")
    for token in doc:
        print(f'|{doc}|{token.text}|{token.lemma_}|{token.pos_}|{token.tag_}|{token.morph.to_dict()}|')

def show_morph_for_sentences_as_markdown_table(sentences):
    sentence_docs = list(nlp.pipe(sentences))
    for sentence_doc in sentence_docs:
        show_morph_as_markdown_table(sentence_doc)

example_sentences = [
    "I was reading the paper",
    "I don’t watch the news, I read the paper",
    "I read the paper yesterday"
]

show_morph_for_sentences_as_markdown_table(example_sentences)