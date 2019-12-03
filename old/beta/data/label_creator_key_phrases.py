"""
    Given domain concepts, output keyword_names.csv (label id and label string) and
    article_keywords.csv (article id and label id)
    Author: Yuren "Rock" Pang and Lily Irvin
    To run spacy, do: python -m spacy download en_core_web_sm
"""
from collections import defaultdict
from gensim import models
from gensim.summarization import keywords, textcleaner
import pandas as pd
import os
import spacy
import gensim
import logging


MAX_PHRASES_PER_ARTICLE = 100  # max phrases retained per article

nlp = spacy.load("en_core_web_sm", disable=['ner', 'tagger'])


def tokenize_sentences(domain_concept, text):
    my_stop_words = ["a", "about", "above", "after", "again", "against", "ain", "all", "am", "an", "and", "any", "are", "aren",
     "aren't", "as", "at", "be", "because", "been", "before", "being", "below", "between", "both", "but", "by", "can",
     "couldn", "couldn't", "d", "did", "didn", "didn't", "do", "does", "doesn", "doesn't", "doing", "don", "don't",
     "down", "during", "each", "few", "for", "from", "further", "had", "hadn", "hadn't", "has", "hasn", "hasn't",
     "have", "haven", "haven't", "having", "he", "her", "here", "hers", "herself", "him", "himself", "his", "how", "i",
     "if", "in", "into", "is", "isn", "isn't", "it", "it's", "its", "itself", "just", "ll", "m", "ma", "me", "mightn",
     "mightn't", "more", "most", "mustn", "mustn't", "my", "myself", "needn", "needn't", "no", "nor", "not", "now", "o",
     "of", "off", "on", "once", "only", "or", "other", "our", "ours", "ourselves", "out", "over", "own", "re", "s",
     "same", "shan", "shan't", "she", "she's", "should", "should've", "shouldn", "shouldn't", "so", "some", "such", "t",
     "than", "that", "that'll", "the", "their", "theirs", "them", "themselves", "then", "there", "these", "they",
     "this", "those", "through", "to", "too", "under", "until", "up", "ve", "very", "was", "wasn", "wasn't", "we",
     "were", "weren", "weren't", "what", "when", "where", "which", "while", "who", "whom", "why", "will", "with", "won",
     "won't", "wouldn", "wouldn't", "y", "you", "you'd", "you'll", "you're", "you've", "your", "yours", "yourself",
     "yourselves", "could", "he'd", "he'll", "he's", "here's", "how's", "i'd", "i'll", "i'm", "i've", "let's", "ought",
     "she'd", "she'll", "that's", "there's", "they'd", "they'll", "they're", "they've", "we'd", "we'll", "we're",
     "we've", "what's", "when's", "where's", "who's", "why's", "would", "able", "abst", "accordance", "according",
     "accordingly", "across", "act", "actually", "added", "adj", "affected", "affecting", "affects", "afterwards", "ah",
     "almost", "alone", "along", "already", "also", "although", "always", "among", "amongst", "announce", "another",
     "anybody", "anyhow", "anymore", "anyone", "anything", "anyway", "anyways", "anywhere", "apparently",
     "approximately", "arent", "arise", "around", "aside", "ask", "asking", "auth", "available", "away", "awfully", "b",
     "back", "became", "become", "becomes", "becoming", "beforehand", "begin", "beginning", "beginnings", "begins",
     "behind", "believe", "beside", "besides", "beyond", "biol", "brief", "briefly", "c", "ca", "came", "cannot",
     "can't", "cause", "causes", "certain", "certainly", "co", "com", "come", "comes", "contain", "containing",
     "contains", "couldnt", "date", "different", "done", "downwards", "due", "e", "ed", "edu", "effect", "eg", "eight",
     "eighty", "either", "else", "elsewhere", "end", "ending", "enough", "especially", "et", "etc", "even", "ever",
     "every", "everybody", "everyone", "everything", "everywhere", "ex", "except", "f", "far", "ff", "fifth", "first",
     "five", "fix", "followed", "following", "follows", "former", "formerly", "forth", "found", "four", "furthermore",
     "g", "gave", "get", "gets", "getting", "give", "given", "gives", "giving", "go", "goes", "gone", "got", "gotten",
     "h", "happens", "hardly", "hed", "hence", "hereafter", "hereby", "herein", "heres", "hereupon", "hes", "hi", "hid",
     "hither", "home", "howbeit", "however", "hundred", "id", "ie", "im", "immediate", "immediately", "importance",
     "important", "inc", "including", "indeed", "index", "information", "instead", "invention", "inward", "itd", "it'll", "j", "k",
     "keep", "keeps", "kept", "kg", "km", "know", "known", "knows", "l", "largely", "last", "lately", "later", "latter",
     "latterly", "least", "less", "lest", "let", "lets", "like", "liked", "likely", "line", "little", "'ll", "look",
     "looking", "looks", "ltd", "made", "mainly", "make", "makes", "many", "may", "maybe", "mean", "means", "meantime",
     "meanwhile", "merely", "mg", "might", "million", "miss", "ml", "moreover", "mostly", "mr", "mrs", "much", "mug",
     "must", "n", "na", "name", "named", "namely", "nay", "nd", "near", "nearly", "necessarily", "necessary", "need", "needs",
     "neither", "never", "nevertheless", "new", "next", "nine", "ninety", "nobody", "non", "none", "nonetheless",
     "noone", "normally", "nos", "noted", "nothing", "nowhere", "obtain", "obtained", "obviously", "often", "oh", "ok",
     "okay", "old", "omitted", "one", "ones", "onto", "ord", "others", "otherwise", "outside", "overall", "owing", "p",
     "page", "pages", "part", "particular", "particularly", "past", "per", "perhaps", "placed", "please", "plus",
     "poorly", "possible", "possibly", "potentially", "pp", "predominantly", "present", "previously", "primarily",
     "probably", "promptly", "proud", "provides", "put", "q", "que", "quickly", "quite", "qv", "r", "ran", "rather",
     "rd", "readily", "really", "recent", "recently", "ref", "refs", "regarding", "regardless", "regards", "related",
     "relatively", "research", "respectively", "resulted", "resulting", "results", "right", "run", "said", "saw", "say",
     "saying", "says", "sec", "section", "see", "seeing", "seem", "seemed", "seeming", "seems", "seen", "self",
     "selves", "sent", "seven", "several", "shall", "shed", "shes", "show", "showed", "shown", "showns", "shows",
     "significant", "significantly", "similar", "similarly", "since", "six", "slightly", "somebody", "somehow",
     "someone", "somethan", "something", "sometime", "sometimes", "somewhat", "somewhere", "soon", "sorry",
     "specifically", "specified", "specify", "specifying", "still", "stop", "strongly", "sub", "substantially",
     "successfully", "sufficiently", "suggest", "sup", "sure", "take", "taken", "taking", "tell", "tends", "th",
     "thank", "thanks", "thanx", "thats", "that've", "thence", "thereafter", "thereby", "thered", "therefore",
     "therein", "there'll", "thereof", "therere", "theres", "thereto", "thereupon", "there've", "theyd", "theyre",
     "think", "thou", "though", "thoughh", "thousand", "throug", "throughout", "thru", "thus", "til", "tip", "together",
     "took", "toward", "towards", "tried", "tries", "truly", "try", "trying", "ts", "twice", "two", "u", "un",
     "unfortunately", "unless", "unlike", "unlikely", "unto", "upon", "ups", "us", "use", "used", "useful", "usefully",
     "usefulness", "uses", "using", "usually", "v", "value", "various", "'ve", "via", "viz", "vol", "vols", "vs", "w",
     "want", "wants", "wasnt", "way", "wed", "welcome", "went", "werent", "whatever", "what'll", "whats", "whence",
     "whenever", "whereafter", "whereas", "whereby", "wherein", "wheres", "whereupon", "wherever", "whether", "whim",
     "whither", "whod", "whoever", "whole", "who'll", "whomever", "whos", "whose", "widely", "willing", "wish",
     "within", "without", "wont", "words", "world", "wouldnt", "www", "x", "yes", "yet", "youd", "youre", "z", "zero",
     "a's", "ain't", "allow", "allows", "apart", "appear", "appreciate", "appropriate", "associated", "best", "better",
     "c'mon", "c's", "cant", "changes", "clearly", "concerning", "consequently", "consider", "considering",
     "corresponding", "course", "currently", "definitely", "described", "despite", "entirely", "exactly", "example",
     "going", "greetings", "hello", "help", "hopefully", "ignored", "inasmuch", "indicate", "indicated", "indicates",
     "inner", "insofar", "it'd", "keep", "keeps", "novel", "presumably", "reasonably", "second", "secondly", "sensible",
     "serious", "seriously", "sure", "t's", "third", "thorough", "thoroughly", "three", "well", "wonder", "a", "about",
     "above", "above", "across", "after", "afterwards", "again", "against", "all", "almost", "alone", "along",
     "already", "also", "although", "always", "am", "among", "amongst", "amoungst", "amount", "an", "and", "another",
     "any", "anyhow", "anyone", "anything", "anyway", "anywhere", "are", "around", "as", "at", "back", "be", "became",
     "because", "become", "becomes", "becoming", "been", "before", "beforehand", "behind", "being", "below", "beside",
     "besides", "between", "beyond", "bill", "both", "bottom", "but", "by", "call", "can", "cannot", "cant", "co",
     "con", "could", "couldnt", "cry", "de", "Des", "describe", "detail", "do", "done", "down", "due", "during", "each", "eg",
     "eight", "either", "eleven", "else", "elsewhere", "empty", "enough", "etc", "even", "ever", "every", "everyone",
     "everything", "everywhere", "except", "few", "fifteen", "fify", "fill", "find", "fire", "first", "five", "for",
     "former", "formerly", "forty", "found", "four", "from", "front", "full", "further", "get", "give", "go", "had",
     "has", "hasnt", "have", "he", "hence", "her", "here", "hereafter", "hereby", "herein", "hereupon", "hers",
     "herself", "him", "himself", "his", "how", "however", "hundred", "ie", "if", "in", "inc", "indeed", "interest",
     "into", "is", "it", "its", "itself", "keep", "last", "latter", "latterly", "least", "less", "ltd", "made", "many",
     "may", "me", "meanwhile", "might", "mill", "mine", "more", "moreover", "most", "mostly", "move", "much", "must",
     "my", "myself", "name", "namely", "neither", "never", "nevertheless", "next", "nine", "no", "nobody", "none",
     "noone", "nor", "not", "nothing", "now", "nowhere", "of", "off", "often", "on", "once", "one", "only", "onto",
     "or", "other", "others", "otherwise", "our", "ours", "ourselves", "out", "over", "own", "part", "per", "perhaps",
     "please", "put", "rather", "re", "same", "see", "seem", "seemed", "seeming", "seems", "serious", "several", "she",
     "should", "show", "side", "since", "sincere", "six", "sixty", "so", "some", "somehow", "someone", "something",
     "sometime", "sometimes", "somewhere", "still", "such", "system", "take", "ten", "than", "that", "the", "their",
     "them", "themselves", "then", "thence", "there", "thereafter", "thereby", "therefore", "therein", "thereupon",
     "these", "they", "thickv", "thin", "third", "this", "those", "though", "three", "through", "throughout", "thru",
     "thus", "to", "together", "too", "top", "toward", "towards", "twelve", "twenty", "two", "un", "under", "until",
     "up", "upon", "us", "very", "via", "was", "we", "well", "were", "what", "whatever", "when", "whence", "whenever",
     "where", "whereafter", "whereas", "whereby", "wherein", "whereupon", "wherever", "whether", "which", "while",
     "whither", "who", "whoever", "whole", "whom", "whose", "why", "will", "with", "within", "without", "would", "yet",
     "you", "your", "yours", "yourself", "yourselves", "the", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k",
     "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H",
     "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "co", "op",
     "research-articl", "pagecount", "cit", "ibid", "les", "le", "au", "que", "est", "pas", "vol", "el", "los", "pp",
     "u201d", "well-b", "http", "volumtype", "par", "0o", "0s", "3a", "3b", "3d", "6b", "6o", "a1", "a2", "a3", "a4",
     "ab", "ac", "ad", "ae", "af", "ag", "aj", "al", "an", "ao", "ap", "ar", "av", "aw", "ax", "ay", "az", "b1", "b2",
     "b3", "ba", "bc", "bd", "be", "bi", "bj", "bk", "bl", "bn", "bp", "br", "bs", "bt", "bu", "bx", "c1", "c2", "c3",
     "cc", "cd", "ce", "cf", "cg", "ch", "ci", "cj", "cl", "cm", "cn", "cp", "cq", "cr", "cs", "ct", "cu", "cv", "cx",
     "cy", "cz", "d2", "da", "dc", "dd", "de", "df", "di", "dj", "dk", "dl", "do", "dp", "dr", "ds", "dt", "du", "dx",
     "dy", "e2", "e3", "ea", "ec", "ed", "ee", "ef", "ei", "ej", "el", "em", "en", "eo", "ep", "eq", "er", "es", "et",
     "eu", "ev", "ex", "ey", "f2", "fa", "fc", "ff", "fi", "fj", "fl", "fn", "fo", "fr", "fs", "ft", "fu", "fy", "ga",
     "ge", "gi", "gj", "gl", "go", "gr", "gs", "gy", "h2", "h3", "hh", "hi", "hj", "ho", "hr", "hs", "hu", "hy", "i",
     "i2", "i3", "i4", "i6", "i7", "i8", "ia", "ib", "ic", "ie", "ig", "ih", "ii", "ij", "il", "in", "io", "ip", "iq",
     "ir", "iv", "ix", "iy", "iz", "jj", "jr", "js", "jt", "ju", "ke", "kg", "kj", "km", "ko", "l2", "la", "lb", "lc",
     "lf", "lj", "ln", "lo", "lr", "ls", "lt", "m2", "ml", "mn", "mo", "ms", "mt", "mu", "n2", "nc", "nd", "ne", "ng",
     "ni", "nj", "nl", "nn", "nr", "ns", "nt", "ny", "oa", "ob", "oc", "od", "of", "og", "oi", "oj", "ol", "om", "on",
     "oo", "oq", "or", "os", "ot", "ou", "ow", "ox", "oz", "p1", "p2", "p3", "pc", "pd", "pe", "pf", "ph", "pi", "pj",
     "pk", "pl", "pm", "pn", "po", "pq", "pr", "ps", "pt", "pu", "py", "qj", "qu", "r2", "ra", "rc", "rd", "rf", "rh",
     "ri", "rj", "rl", "rm", "rn", "ro", "rq", "rr", "rs", "rt", "ru", "rv", "ry", "s2", "sa", "sc", "sd", "se", "sf",
     "si", "sj", "sl", "sm", "sn", "sp", "sq", "sr", "ss", "st", "sy", "sz", "t1", "t2", "t3", "tb", "tc", "td", "te",
     "tf", "th", "ti", "tj", "tl", "tm", "tn", "tp", "tq", "tr", "ts", "tt", "tv", "tx", "ue", "ui", "uj", "uk", "um",
     "un", "uo", "ur", "ut", "va", "wa", "vd", "wi", "vj", "vo", "wo", "vq", "vt", "vu", "x1", "x2", "x3", "xf", "xi",
     "xj", "xk", "xl", "xn", "xo", "xs", "xt", "xv", "xx", "y2", "yj", "yl", "yr", "ys", "yt", "zi", "zz"]

    for stop_word in my_stop_words:
        lexeme = nlp.vocab[stop_word]
        lexeme.is_stop = True

    sentences = []

    if not isinstance(text, float):
        text_sentences = textcleaner.split_sentences(text)
        cleaned_sentences = []

        for sent in text_sentences:
            cleaned_sentences.append(nlp(sent.lower()))

        for sentence in cleaned_sentences:
            sent = []
            for w in sentence:
                if (not w.is_stop) and (not w.is_punct) and (not w.like_num) and (not w.like_url) and \
                        ('\n' not in w.text) and (' ' not in w.text) and (len(w.text) > 1):
                    sent.append(w.text.strip())
            sentences.append(sent)
    else:
        logging.warning(domain_concept + ": does not have summary")
        pass

    return sentences


def phrasify(article_text):
    """
    Given the summary of each domain concept, remove stop words, punctuations, numbers, and newline,
    Use gensim to identify phrases within the text. Returns a list of documents, where each document
    is a list of phrases in the document.
    """
    corpus = []

    for row in article_text.itertuples():
        sentences = tokenize_sentences(row.article_name, row.text)
        corpus.extend([(row.article_id, row.article_name, s) for s in sentences])
    sentences = [doc[-1] for doc in corpus]

    bigram = gensim.models.Phrases(sentences, min_count=30)
    phrases1 = list(bigram[sentences])
    trigram = gensim.models.Phrases(phrases1, min_count=10)
    phrases2 = list(trigram[phrases1])

    assert len(phrases2) == len(phrases1) == len(corpus)

    phrase_corpus = [(doc[0], doc[1], sentence) for (doc, sentence) in zip(corpus, phrases2)]

    return phrase_corpus


def fetch_key_phrases(domain_concept, text, bigram, trigram):
    key_phrases = set()
    sentences = tokenize_sentences(domain_concept, text)

    for sentence in sentences:
        bigrams = bigram[sentence]
        trigrams = trigram[bigrams]
        key_phrases.update(bigrams + trigrams)

    my_stop_words = ['say', '\'s', 'be', 'says', 'including', 'said', 'named', '\t', 'know', '\n\n', 'Des', ' ']
    for stopword in my_stop_words:
        lexeme = nlp.vocab[stopword]
        lexeme.is_stop = True

    if not isinstance(text, float):
        sum = nlp(text)
        cleaned_summary = ''
        for w in sum:
            if w.text != '\n' and not w.is_stop and not w.is_punct and not w.like_num and len(w.text) > 1:
                    cleaned_summary += ' ' + w.text
        for word in keywords(cleaned_summary, lemmatize=True).split('\n'):
            key_phrases.add(word)

    return key_phrases


def create_labels(phrase_corpus):
    """
    Find the text of each domain concept and creates a data frame with articles and keyword labels
    :return: a dataframe with article id and label id
    """

    article_id_to_name = {}
    article_word_freq = defaultdict(lambda: defaultdict(float))  # article id -> word id -> freq
    vocab = {}  # word -> word id
    for article_id, article_name, article_text in phrase_corpus:
        article_id_to_name[article_id] = article_name
        for word in article_text:
            if word not in vocab:
                vocab[word] = len(vocab)
            word_id = vocab[word]
            article_word_freq[article_id][word_id] += 1

    sparse_matrix = []
    for article_id in sorted(article_word_freq.keys()):
        sparse_row = []
        for word_id in sorted(article_word_freq[article_id].keys()):
            sparse_row.append((word_id, article_word_freq[article_id][word_id]))
        sparse_matrix.append((article_id, article_id_to_name[article_id], sparse_row))

    index2word = {i : w for (w, i) in vocab.items()}
    tfidf = models.TfidfModel([row for (id, name, row) in sparse_matrix])
    results = []
    for id, name, row in sparse_matrix:
        tfidf_row = tfidf[row]
        tfidf_row.sort(key=lambda x: x[1], reverse=True)
        results.append((id, name, tfidf_row[:MAX_PHRASES_PER_ARTICLE]))

    return index2word, results


def create_label_id_str_csv(directory, index2word):
    id_to_label = list(index2word.items())
    labels_df = pd.DataFrame(id_to_label, columns=["label_id", "label"])
    labels_df.to_csv(directory + '/keyphrases_names.csv', index=False)


def create_article_label_csv(tf_idf_results, directory):
    rows = []
    for (article_id, article_name, tfidf) in tf_idf_results:
        for (word_id, score) in tfidf:
            rows.append({'article_id' : article_id, 'label_id' : word_id, 'score' : score})
    df = pd.DataFrame(rows)
    df.to_csv(directory + "/article_keyphrases.csv", index=False)


def main(map_directory):
    if not os.path.exists(map_directory):
        os.makedirs(map_directory)

    article_text = pd.read_csv(map_directory + '/article_text_gloss.csv')
    phrase_corpus = phrasify(article_text)
    index2word, tf_idf_results = create_labels(phrase_corpus)
    create_article_label_csv(tf_idf_results, map_directory)
    create_label_id_str_csv(map_directory, index2word)


if __name__ == '__main__':
    import sys

    if len(sys.argv) != 2:
        sys.stderr.write('Usage: %s map_directory' % sys.argv[0])
        sys.exit(1)

    map_directory = sys.argv[1]
    main(map_directory)
