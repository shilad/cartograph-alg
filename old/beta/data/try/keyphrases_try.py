# # import nltk
# # # from nltk.tokenize import word_tokenize
# # # from nltk.tag import pos_tag
# # #
ex = 'European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices'
# # # def preprocess(sent):
# # #     sent = nltk.word_tokenize(sent)
# # #     sent = nltk.pos_tag(sent)
# # #     return sent
# # #
# # # sent = preprocess(ex)
# # #
# # #
# # # pattern = 'NP: {<DT>?<JJ>*<NN>}'
# # # cp = nltk.RegexpParser(pattern)
# # # cs = cp.parse(sent)
# # #
# # #
# # # from nltk.chunk import conlltags2tree, tree2conlltags
# # # from pprint import pprint
# # # iob_tagged = tree2conlltags(cs)
# # # pprint(iob_tagged)
# # #
# # # ne_tree = nltk.ne_chunk(pos_tag(word_tokenize(ex)))
# # # print(ne_tree)
#
# # import spacy
# # from spacy import displacy
# # from collections import Counter
# # import en_core_web_sm
# # from pprint import pprint
# # nlp = en_core_web_sm.load()
# #
# # # doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
# # # pprint([(X.text, X.label_) for X in doc.ents])
# # # pprint([(X, X.ent_iob_, X.ent_type_) for X in doc])
# #
# # # text = """ambohimanga is a hill and traditional fortified royal settlement (rova) in madagascar, located approximately 24 kilometers (15 mi) northeast of the capital city of antananarivo. the hill and the rova that stands on top are considered the most significant symbol of the cultural identity of the merina people and the most important and best-preserved monument of the precolonial merina kingdom. the walled historic village includes residences and burial sites of several key monarchs. the site, one of the twelve sacred hills of imerina, is associated with strong feelings of national identity and has maintained its spiritual and sacred character both in ritual practice and the popular imagination for at least four hundred years. it remains a place of worship to which pilgrims come from madagascar and elsewhere.
# # # the site has been politically important since the early 18th century, when king andriamasinavalona (1675–1710) divided the kingdom of imerina into four quadrants and assigned his son andriantsimitoviaminiandriana to govern the northeastern quadrant, avaradrano, from its newly designated capital at ambohimanga. the division of imerina led to 77 years of civil war, during which time the successive rulers of avaradrano led military campaigns to expand their territory while undertaking modifications to the defenses at ambohimanga to better protect it against attacks. the war was ended from ambohimanga by king andrianampoinimerina, who successfully undertook negotiations and military campaigns that reunited imerina under his rule by 1793. upon capturing the historic capital of imerina at antananarivo, andrianampoinimerina shifted his royal court and all political functions back to its original locus at antananarivo's royal compound and declared the two cities of equal importance, with ambohimanga as the kingdom's spiritual capital. he and later rulers in his line continued to conduct royal rituals at the site and regularly inhabited and remodeled ambohimanga until french colonization of the kingdom and the exile of the royal family in 1897. the significance of historical events here and the presence of royal tombs have given the hill a sacred character that is further enhanced at ambohimanga by the burial sites of several vazimba, the island's earliest inhabitants.
# # # the royal compound on the hilltop is surrounded by a complex system of defensive ditches and stone walls and is accessed by 14 gateways, of which many were sealed by stone disc barriers. the gateways and construction of buildings within the compound are arranged according to two overlaid cosmological systems that value the four cardinal points radiating from a unifying center, and attach sacred importance to the northeastern direction. the complex inside the wall is subdivided into three smaller rova. mahandrihono, the largest compound, was established between 1710 and 1730 by king andriambelomasina; it remains largely intact and contains the royal tombs, house of king andrianampoinimerina, summer palace of queen ranavalona ii, and sites that figured in key royal rituals such as the sacrificial zebu pen, royal bath and main courtyard. original buildings no longer remain in the compound of bevato, established before 1710 by andriamborona, and the nanjakana compound, built for king andrianjafy in the late 19th century. the hill and its royal fortified city were added to the list of unesco world heritage sites in 2001 and represent madagascar's only cultural site following the destruction by fire in 1995 of its historic sister city, the rova of antananarivo, shortly before the latter's intended inscription to the list. numerous governmental and civil society organizations support the conservation of ambohimanga by restoring damaged features and preventing further degradation."""
# # #
# # # article = nlp(text)[4].left_edge.i
# # # print(article)
# #
# # nlp = spacy.load("en_core_web_sm")
# # doc = nlp("Credit and mortgage account holders must submit their requests")
# # span = doc[doc[4].left_edge.i : doc[4].right_edge.i+1]
# # print(span)
# # with doc.retokenize() as retokenizer:
# #     retokenizer.merge(span)
# # for token in doc:
# #     print(token.text, token.pos_, token.dep_, token.head.text)
#
# from nltk.corpus import wordnet as wn
#
# # Just to make it a bit more readable
# WN_NOUN = 'n'
# WN_VERB = 'v'
# WN_ADJECTIVE = 'a'
# WN_ADJECTIVE_SATELLITE = 's'
# WN_ADVERB = 'r'
#
#
# def convert(word, from_pos, to_pos):
#     """ Transform words given from/to POS tags """
#
#     synsets = wn.synsets(word, pos=from_pos)
#
#     # Word not found
#     if not synsets:
#         return []
#
#     # Get all lemmas of the word (consider 'a'and 's' equivalent)
#     lemmas = [l for s in synsets
#               for l in s.lemmas
#               if s.name.split('.')[1] == from_pos
#               or from_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)
#               and s.name.split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)]
#
#     # Get related forms
#     derivationally_related_forms = [(l, l.derivationally_related_forms()) for l in lemmas]
#
#     # filter only the desired pos (consider 'a' and 's' equivalent)
#     related_noun_lemmas = [l for drf in derivationally_related_forms
#                            for l in drf[1]
#                            if l.synset.name.split('.')[1] == to_pos
#                            or to_pos in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)
#                            and l.synset.name.split('.')[1] in (WN_ADJECTIVE, WN_ADJECTIVE_SATELLITE)]
#
#     # Extract the words from the lemmas
#     words = [l.name for l in related_noun_lemmas]
#     len_words = len(words)
#
#     # Build the result in the form of a list containing tuples (word, probability)
#     result = [(w, float(words.count(w)) / len_words) for w in set(words)]
#     result.sort(key=lambda w: -w[1])
#
#     # return all the possibilities sorted by probability
#     return result
#
CONTRACTION_MAP = {
"ain't": "is not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how is",
"I'd": "I would",
"I'd've": "I would have",
"I'll": "I will",
"I'll've": "I will have",
"I'm": "I am",
"I've": "I have",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so as",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
"we'd": "we would",
"we'd've": "we would have",
"we'll": "we will",
"we'll've": "we will have",
"we're": "we are",
"we've": "we have",
"weren't": "were not",
"what'll": "what will",
"what'll've": "what will have",
"what're": "what are",
"what's": "what is",
"what've": "what have",
"when's": "when is",
"when've": "when have",
"where'd": "where did",
"where's": "where is",
"where've": "where have",
"who'll": "who will",
"who'll've": "who will have",
"who's": "who is",
"who've": "who have",
"why's": "why is",
"why've": "why have",
"will've": "will have",
"won't": "will not",
"won't've": "will not have",
"would've": "would have",
"wouldn't": "would not",
"wouldn't've": "would not have",
"y'all": "you all",
"y'all'd": "you all would",
"y'all'd've": "you all would have",
"y'all're": "you all are",
"y'all've": "you all have",
"you'd": "you would",
"you'd've": "you would have",
"you'll": "you will",
"you'll've": "you will have",
"you're": "you are",
"you've": "you have"
}
import spacy
import re
import nltk
import en_core_web_sm
import unicodedata
from nltk.tokenize.toktok import ToktokTokenizer

nlp = en_core_web_sm.load()

doc = nlp('European authorities fined Google a record $5.1 billion on Wednesday for abusing its power in the mobile phone market and ordered the company to alter its practices')
def remove_accented_chars(text):
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    return text


def expand_contractions(text, contraction_mapping=CONTRACTION_MAP):
    contractions_pattern = re.compile('({})'.format('|'.join(contraction_mapping.keys())),
                                      flags=re.IGNORECASE | re.DOTALL)

    def expand_match(contraction):
        match = contraction.group(0)
        first_char = match[0]
        expanded_contraction = contraction_mapping.get(match) \
            if contraction_mapping.get(match) \
            else contraction_mapping.get(match.lower())
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    expanded_text = contractions_pattern.sub(expand_match, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text

def remove_special_characters(text, remove_digits=False):
    REGEX_STRIP_LATEXT = r"(\n(?=(\s+))[^a-z]+(?=((?<!(\.|:))\n)))|{.+}|\b\w{1,1}\b|(==.+==)|\ssin|\scos|\stan|\(.+\)"
    REGEX_STRIP_UNICODE_SPACE = r"\w*[\\]\w*"
    text = re.sub(REGEX_STRIP_LATEXT, "", text)
    text = re.sub(REGEX_STRIP_UNICODE_SPACE, "", text).strip("\"")

    pattern = r'[^a-zA-z0-9\s]' if not remove_digits else r'[^a-zA-z\s]'
    text = re.sub(pattern, '', text)
    return text

def lemmatize_text(text):
    text = nlp(text)
    text = ' '.join([word.lemma_ if word.lemma_ != '-PRON-' else word.text for word in text])
    return text

def remove_stopwords(text, is_lower_case=False):
    tokenizer = ToktokTokenizer()
    stopword_list = nltk.corpus.stopwords.words('english')
    stopword_list.remove('no')
    stopword_list.remove('not')

    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    if is_lower_case:
        filtered_tokens = [token for token in tokens if token not in stopword_list]
    else:
        filtered_tokens = [token for token in tokens if token.lower() not in stopword_list]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def normalize_corpus(corpus, contraction_expansion=True,
                     accented_char_removal=True, text_lower_case=True,
                     text_lemmatization=True, special_char_removal=True,
                     stopword_removal=True, remove_digits=True):
    normalized_corpus = []
    # normalize each document in the corpus
    for doc in corpus:
        # remove accented characters
        if accented_char_removal:
            doc = remove_accented_chars(doc)
        # expand contractions
        if contraction_expansion:
            doc = expand_contractions(doc)
        # lowercase the text
        if text_lower_case:
            doc = doc.lower()
        # remove extra newlines
        doc = re.sub(r'[\r|\n|\r\n]+', ' ', doc)
        # lemmatize text
        if text_lemmatization:
            doc = lemmatize_text(doc)
        # remove special characters and\or digits
        if special_char_removal:
            # insert spaces between special characters to isolate them
            special_char_pattern = re.compile(r'([{.(-)!}])')
            doc = special_char_pattern.sub(" \\1 ", doc)
            doc = remove_special_characters(doc, remove_digits=remove_digits)
            # remove extra whitespace
        doc = re.sub(' +', ' ', doc)
        # remove stopwords
        if stopword_removal:
            doc = remove_stopwords(doc, is_lower_case=text_lower_case)

        normalized_corpus.append(doc)

    return normalized_corpus
#
text = """antarctica (uk:  or , us:  (listen)) is earth's southernmost continent. it contains the geographic south pole and is situated in the antarctic region of the southern hemisphere, almost entirely south of the antarctic circle, and is surrounded by the southern ocean. at 14,200,000 square kilometres (5,500,000 square miles), it is the fifth-largest continent and nearly twice the size of australia. at 0.00008 people per square kilometre, it is by far the least densely populated continent. about 98% of antarctica is covered by ice that averages 1.9 km (1.2 mi; 6,200 ft) in thickness, which extends to all but the northernmost reaches of the antarctic peninsula.
antarctica, on average, is the coldest, driest, and windiest continent, and has the highest average elevation of all the continents. most of antarctica is a polar desert, with annual precipitation of 20 cm (7.9 in) along the coast and far less inland; there has been no rain here for almost 2 million years, yet 80% of the world freshwater reserves are stored here. the temperature in antarctica has reached −89.2 °c (−128.6 °f) (or even −94.7 °c (−135.8 °f) as measured from space), though the average for the third quarter (the coldest part of the year) is −63 °c (−81 °f). anywhere from 1,000 to 5,000 people reside throughout the year at research stations scattered across the continent. organisms native to antarctica include many types of algae, bacteria, fungi, plants, protista, and certain animals, such as mites, nematodes, penguins, seals and tardigrades. vegetation, where it occurs, is tundra.
antarctica is noted as the last region on earth in recorded history to be discovered, unseen until 1820 when the russian expedition of fabian gottlieb von bellingshausen and mikhail lazarev on vostok and mirny sighted the fimbul ice shelf. the continent, however, remained largely neglected for the rest of the 19th century because of its hostile environment, lack of easily accessible resources, and isolation. in 1895, the first confirmed landing was conducted by a team of norwegians.
antarctica is a de facto condominium, governed by parties to the antarctic treaty system that have consulting status. twelve countries signed the antarctic treaty in 1959, and thirty-eight have signed it since then. the treaty prohibits military activities and mineral mining, prohibits nuclear explosions and nuclear waste disposal, supports scientific research, and protects the continent's ecozone. ongoing experiments are conducted by more than 4,000 scientists from many nations."""
#
# #
# # from collections import Counter
# # article = """western asia, west asia, southwestern asia or southwest asia is the westernmost subregion of asia. the concept is in limited use, as it significantly overlaps with the middle east (or the near east), the main difference usually being the exclusion of the majority of egypt, which would be counted as part of north africa (that is, egypt without the sinai peninsula), and of european turkey and the inclusion of the caucasus. the term is sometimes used for the purposes of grouping countries in statistics, in which case egypt might be excluded and turkey included entirely. the total population of western asia is an estimated 300 million as of 2015. although the term ""western asia"" is mostly used as a convenient division of contemporary sovereign states into a manageable number of world regions for statistical purposes, it is sometimes used instead of the more geopolitical term ""middle east\""""
# # article = nlp(article)
# # for token in article:
# #     print(token.text, token.pos_, token.dep_, token.head.text)
#
# import pke, string
# from nltk.corpus import stopwords
# extractor = pke.supervised.Kea()
# extractor.load_document(text, language='en')
#
# # 3. select {1-3}-grams not containing punctuation marks and not
# #    beginning/ending with a stopword as candidates.
# stoplist = stopwords.words('english')
# stoplist.append("approximately")
import pke
import pandas as pd
# # pke.compute_document_frequency("/Users/senresearchlab/PycharmProjects/cartograph-alg/data/georgraphy/article_text_gloss.csv", "./doc_frequency.tsv.gz")
labels_to_id = {}
rows_lst = []
article = pd.read_csv("/Users/senresearchlab/PycharmProjects/cartograph-alg/data/georgraphy/article_text_gloss.csv")
for row in article.itertuples():
    text = row.text
    if not isinstance(text, float):
        extractor = pke.unsupervised.KPMiner()
        extractor.load_document(text, language='en', normalization=None)

        lasf = 4
        cutoff = 200
        extractor.candidate_selection(lasf=lasf, cutoff=cutoff)

        df = pke.load_document_frequency_file(input_file='./doc_frequency.tsv.gz')
        alpha = 2.3
        sigma = 3.0
        extractor.candidate_weighting(df=df, alpha=alpha, sigma=sigma)

        # 5. get the 10-highest scored candidates as keyphrases
        keyphrases = extractor.get_n_best(n=10)
        if len(keyphrases) is not 0:
            for keyphrase, score in keyphrases:
                if keyphrase not in labels_to_id:
                    labels_to_id[keyphrase] = len(labels_to_id)
                id = labels_to_id.get(keyphrase, len(labels_to_id))
                rows_lst.append(
                    {"article_id": row.article_id, "label_id": id, "score": score})

pd.DataFrame(rows_lst).to_csv("article_keyphrases.csv", index=False)
id_to_label = [(id, label) for (label, id) in labels_to_id.items()]
labels_df = pd.DataFrame(id_to_label, columns=["label_id", "label"])
labels_df.to_csv('keyphrases_names.csv', index=False)
