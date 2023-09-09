# pip install transformers
# pip install transformers[tf-cpu] - to use tensorflow
# pip install torch - to load weights into TFAutoModelForTokenClassification from torch
from transformers import AutoTokenizer, TFAutoModelForTokenClassification, TokenClassificationPipeline

def get_pos_pipeline():
    model_name = "QCRI/bert-base-multilingual-cased-pos-english"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = TFAutoModelForTokenClassification.from_pretrained(model_name, from_pt=True)
    return TokenClassificationPipeline(model=model, tokenizer=tokenizer)

def tag_sentence(pipeline, sentence):
    outputs = pipeline(sentence)
    print(outputs)

# only need to get the pipeline once
pipeline = get_pos_pipeline()

tag_sentence(pipeline, "This is the first sentence of the text")
tag_sentence(pipeline, "NLTK is good for POS tagging but transformers are better")
