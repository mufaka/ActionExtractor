# for huggingface 
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

'''
[
    {'entity': 'DT', 'score': 0.99975103, 'index': 1, 'word': 'This', 'start': 0, 'end': 4}, 
    {'entity': 'VBZ', 'score': 0.9995925, 'index': 2, 'word': 'is', 'start': 5, 'end': 7}, 
    {'entity': 'DT', 'score': 0.99982387, 'index': 3, 'word': 'the', 'start': 8, 'end': 11}, 
    {'entity': 'JJ', 'score': 0.99956053, 'index': 4, 'word': 'first', 'start': 12, 'end': 17}, 
    {'entity': 'NN', 'score': 0.99977154, 'index': 5, 'word': 'sentence', 'start': 18, 'end': 26}, 
    {'entity': 'IN', 'score': 0.9998235, 'index': 6, 'word': 'of', 'start': 27, 'end': 29}, 
    {'entity': 'DT', 'score': 0.9998578, 'index': 7, 'word': 'the', 'start': 30, 'end': 33}, 
    {'entity': 'NN', 'score': 0.9997501, 'index': 8, 'word': 'text', 'start': 34, 'end': 38}
]
'''
tag_sentence(pipeline, "This is the first sentence of the text.")

'''
[
    {'entity': 'NNP', 'score': 0.99833566, 'index': 1, 'word': 'NL', 'start': 0, 'end': 2}, 
    {'entity': 'NNP', 'score': 0.99889416, 'index': 2, 'word': '##T', 'start': 2, 'end': 3}, 
    {'entity': 'NNP', 'score': 0.99762017, 'index': 3, 'word': '##K', 'start': 3, 'end': 4}, 
    {'entity': 'VBZ', 'score': 0.9996793, 'index': 4, 'word': 'is', 'start': 5, 'end': 7}, 
    {'entity': 'JJ', 'score': 0.9986487, 'index': 5, 'word': 'good', 'start': 8, 'end': 12}, 
    {'entity': 'IN', 'score': 0.9998136, 'index': 6, 'word': 'for', 'start': 13, 'end': 16}, 
    {'entity': 'NNP', 'score': 0.903765, 'index': 7, 'word': 'P', 'start': 17, 'end': 18}, 
    {'entity': 'NNS', 'score': 0.35405183, 'index': 8, 'word': '##OS', 'start': 18, 'end': 20}, 
    {'entity': 'VBG', 'score': 0.514822, 'index': 9, 'word': 'tag', 'start': 21, 'end': 24}, 
    {'entity': 'NN', 'score': 0.5284631, 'index': 10, 'word': '##ging', 'start': 24, 'end': 28}, 
    {'entity': 'CC', 'score': 0.9994684, 'index': 11, 'word': 'but', 'start': 29, 'end': 32}, 
    {'entity': 'NNS', 'score': 0.9991509, 'index': 12, 'word': 'transform', 'start': 33, 'end': 42}, 
    {'entity': 'NNS', 'score': 0.99885714, 'index': 13, 'word': '##ers', 'start': 42, 'end': 45}, 
    {'entity': 'VBP', 'score': 0.99942386, 'index': 14, 'word': 'are', 'start': 46, 'end': 49}, 
    {'entity': 'JJR', 'score': 0.9486666, 'index': 15, 'word': 'better', 'start': 50, 'end': 56}
]
'''
tag_sentence(pipeline, "NLTK is good for POS tagging but transformers are better.")

'''
[
    {'entity': 'PRP', 'score': 0.9997768, 'index': 1, 'word': 'You', 'start': 0, 'end': 3}, 
    {'entity': 'MD', 'score': 0.99962735, 'index': 2, 'word': 'should', 'start': 4, 'end': 10}, 
    {'entity': 'VB', 'score': 0.9996045, 'index': 3, 'word': 'do', 'start': 11, 'end': 13}, 
    {'entity': 'PRP$', 'score': 0.9997112, 'index': 4, 'word': 'your', 'start': 14, 'end': 18}, 
    {'entity': 'NN', 'score': 0.99970275, 'index': 5, 'word': 'home', 'start': 19, 'end': 23}, 
    {'entity': 'NN', 'score': 0.9995745, 'index': 6, 'word': '##work', 'start': 23, 'end': 27}, 
    {'entity': '.', 'score': 0.99991775, 'index': 7, 'word': '.', 'start': 27, 'end': 28}
]
'''
tag_sentence(pipeline, "You should do your homework.")

'''
[
    {'entity': 'VB', 'score': 0.9885771, 'index': 1, 'word': 'Ad', 'start': 0, 'end': 2}, 
    {'entity': 'VB', 'score': 0.9911873, 'index': 2, 'word': '##d', 'start': 2, 'end': 3}, 
    {'entity': 'DT', 'score': 0.999871, 'index': 3, 'word': 'a', 'start': 4, 'end': 5}, 
    {'entity': 'NN', 'score': 0.99943393, 'index': 4, 'word': 'modi', 'start': 6, 'end': 10}, 
    {'entity': 'NN', 'score': 0.9995258, 'index': 5, 'word': '##fier', 'start': 10, 'end': 14}, 
    {'entity': 'IN', 'score': 0.99983084, 'index': 6, 'word': 'of', 'start': 15, 'end': 17}, 
    {'entity': 'DT', 'score': 0.6283519, 'index': 7, 'word': 'either', 'start': 18, 'end': 24}, 
    {'entity': 'NNP', 'score': 0.9907198, 'index': 8, 'word': 'RT', 'start': 25, 'end': 27}, 
    {'entity': 'CC', 'score': 0.9997769, 'index': 9, 'word': 'or', 'start': 28, 'end': 30}, 
    {'entity': 'NNP', 'score': 0.99415576, 'index': 10, 'word': 'L', 'start': 31, 'end': 32}, 
    {'entity': 'NNP', 'score': 0.992155, 'index': 11, 'word': '##T', 'start': 32, 'end': 33}, 
    {'entity': 'TO', 'score': 0.99986815, 'index': 12, 'word': 'to', 'start': 34, 'end': 36}, 
    {'entity': 'NN', 'score': 0.98477006, 'index': 13, 'word': 'procedure', 'start': 37, 'end': 46}, 
    {'entity': 'CD', 'score': 0.997207, 'index': 14, 'word': '308', 'start': 47, 'end': 50}, 
    {'entity': 'CD', 'score': 0.99928015, 'index': 15, 'word': '##01', 'start': 50, 'end': 52}, 
    {'entity': 'CC', 'score': 0.99981135, 'index': 16, 'word': 'and', 'start': 53, 'end': 56}, 
    {'entity': 'RB', 'score': 0.99868983, 'index': 17, 'word': 'then', 'start': 57, 'end': 61}, 
    {'entity': 'VB', 'score': 0.99499327, 'index': 18, 'word': 'res', 'start': 62, 'end': 65}, 
    {'entity': 'VB', 'score': 0.9950729, 'index': 19, 'word': '##ub', 'start': 65, 'end': 67}, 
    {'entity': 'VB', 'score': 0.9943663, 'index': 20, 'word': '##mit', 'start': 67, 'end': 70}, 
    {'entity': 'DT', 'score': 0.99987495, 'index': 21, 'word': 'the', 'start': 71, 'end': 74}, 
    {'entity': 'NN', 'score': 0.99983776, 'index': 22, 'word': 'claim', 'start': 75, 'end': 80}, 
    {'entity': '.', 'score': 0.9999225, 'index': 23, 'word': '.', 'start': 80, 'end': 81}
]
'''
tag_sentence(pipeline, "Add a modifier of either RT or LT to procedure 30801 and then resubmit the claim.")

