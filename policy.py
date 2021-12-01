import requests
import re
import stanza
from bs4 import BeautifulSoup
from nltk.corpus import stopwords

nlp = stanza.Pipeline('en', processors={'ner': 'CoNLL03'})
stop_words = set(stopwords.words('english'))

# parses the given policy html
def parse_html(html):
    parse_map = {}
    
    skip_tags = set({'script', 'style', 'form', 'nav', 'footer', 'ul', 'li', 'a', 'br', 'hr', 'ol'})
    count = 0 
    features_str = 'Count,Tag,Signature,X1,X2\n'

    for element in html.body.select('*'):
        tag_name = element.name.lower().strip()

        clean_text = element.text.lower().strip().replace('\n', ' ')
        clean_text = re.sub(r'\s+', ' ', clean_text)

        if tag_name in skip_tags or len(clean_text) < 4:
            continue

        count += 1

        key = '{text}|-|{count}'.format(text=clean_text, count=count)
        signature = get_parents(element)
        parse_map[key] = signature

        features_str += gen_feature_row(element, count, tag_name, clean_text, signature)

    with open('features.csv', 'w') as features_file:
        features_file.write(features_str)

    return parse_map


# generate the feature row for the current element
def gen_feature_row(element, count, tag_name, clean_text, signature):
    features = get_features(element.text)

    return '{count},{tag_name},{signature},{text_length},{features}\n'.format(
        count=count,
        tag_name=tag_name,
        signature=signature,
        text_length=len(clean_text),
        features=features
    )


# returns a string containing all of the parent tags of the given element
def get_parents(element):
    signature = ''

    for parent in element.find_all_previous():
        parent_tag = parent.name.lower().strip()

        if parent_tag == 'body':
            continue

        signature = signature + ' ' + parent_tag 

    return signature.strip()


# get the total number of features in the text
def get_features(text):
    num_special_chars = len(re.sub('[\w]+', '', text))
    num_words = len(text.split(' '))
    num_stop_words = 0

    for word in text.split():
        if word in stop_words:
            num_stop_words += 1

    num_sentences = len(text.split('.'))
    num_ner_slots = get_ner_slots(text) 
    num_titles = sum(map(str.istitle, text.split()))

    return num_special_chars + num_words + num_stop_words + num_sentences + num_ner_slots + num_titles


# get the number of ner tags in the given text
def get_ner_slots(text):
    count = 0
    doc = nlp(text)

    for sentence in doc.sentences:
        for token in sentence.tokens:
            if token.ner != 'O':
                count += 1

    return count 


if __name__ == '__main__':
    r = requests.get('https://www.apple.com/legal/privacy/en-ww/')
    html = BeautifulSoup(r.content, 'html.parser')

    parse_html(html)

