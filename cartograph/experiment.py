import spacy
import string
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English

# Create our list of punctuation marks
punctuations = string.punctuation

# Create our list of stopwords
nlp = spacy.load('en')
stop_words = spacy.lang.en.stop_words.STOP_WORDS

# Load English tokenizer, tagger, parser, NER and word vectors
parser = English()

# Creating our tokenizer function
def spacy_tokenizer(sentence):
    # Creating our token object, which is used to create documents with linguistic annotations.
    mytokens = parser(sentence)

    # Lemmatizing each token and converting each token into lowercase
    mytokens = [ word.lemma_.lower().strip() if word.lemma_ != "-PRON-" else word.lower_ for word in mytokens ]

    # Removing stop words
    mytokens = [ word for word in mytokens if word not in stop_words and word not in punctuations ]

    # return preprocessed list of tokens
    return mytokens

sentence = '''Acetic acid , systematically named ethanoic acid , is a colourless liquid organic compound with the chemical formula CH3COOH (also written as CH3CO2H or C2H4O2). When undiluted, it is sometimes called glacial acetic acid. Vinegar is no less than 4% acetic acid by volume, making acetic acid the main component of vinegar apart from water. Acetic acid has a distinctive sour taste and pungent smell. In addition to household vinegar, it is mainly produced as a precursor to polyvinyl acetate and cellulose acetate. It is classified as a weak acid since it only partially dissociates in solution, but concentrated acetic acid is corrosive and can attack the skin.
Acetic acid is the second simplest carboxylic acid (after formic acid). It consists of a methyl group attached to a carboxyl group. It is an important chemical reagent and industrial chemical, used primarily in the production of cellulose acetate for photographic film, polyvinyl acetate for wood glue, and synthetic fibres and fabrics. In households, diluted acetic acid is often used in descaling agents. In the food industry, acetic acid is controlled by the food additive code E260 as an acidity regulator and as a condiment. In biochemistry, the acetyl group, derived from acetic acid, is fundamental to all forms of life. When bound to coenzyme A, it is central to the metabolism of carbohydrates and fats.
The global demand for acetic acid is about 6.5 million metric tons per year (Mt/a), of which approximately 1.5 Mt/a is met by recycling; the remainder is manufactured from methanol. Vinegar is mostly dilute acetic acid, often produced by fermentation and subsequent oxidation of ethanol."
1,Alcohol laws of New Jersey,"The state laws governing alcoholic drinks in New Jersey are among the most complex in the United States, with many peculiarities not found in other states' laws. They provide for 29 distinct liquor licenses granted to manufacturers, wholesalers, retailers, and for the public warehousing and transport of alcoholic drinks. General authority for the statutory and regulatory control of alcoholic drinks rests with the state government, particularly the Division of Alcoholic Beverage Control overseen by the state's Attorney General.
Under home rule, New Jersey law grants individual municipalities substantial discretion in passing ordinances regulating the sale and consumption of alcoholic drinks within their limits. The number of retail licenses available is determined by a municipality's population, and may be further limited by the town's governing body. As a result, the availability of alcohol and regulations governing it vary significantly from town to town. A small percentage of municipalities in the state are ""dry towns"" that do not allow alcoholic drinks to be sold, and do not issue retail licenses for bars or restaurants to serve alcohol to patrons. Other towns permit alcohol sales 24 hours a day. Retail licenses tend to be difficult to obtain, and when available are subject to exorbitant prices and fervent competition.
In addition to granting local governments wide latitude over liquor sales, New Jersey law has some other unusual features. Corporations are limited to two retail distribution licenses, making it impractical for chain stores to sell alcoholic drinks; this restriction, in conjunction with municipal ordinances, severely limits supermarket and convenience store chains from selling beer as they do in many other states. State law treats drunk driving as a traffic offense rather than a crime, and permits individual municipalities to define the scope of underage drinking laws."
2,American paddlefish,"The American paddlefish (Polyodon spathula) is a species of basal ray-finned fish closely related to sturgeons in the order Acipenseriformes. Fossil records of paddlefish date back over 300 million years, nearly 50 million years before dinosaurs first appeared.  American paddlefish are smooth-skinned freshwater fish commonly called paddlefish, but are also referred to as Mississippi paddlefish, spoon-billed cats, or spoonbills.  They are one of only two extant species in the paddlefish family, Polyodontidae. The other is the critically endangered Chinese paddlefish (Psephurus gladius) endemic to the Yangtze River basin in China.  American paddlefish are often referred to as primitive fish, or relict species because they retain some morphological characteristics of their early ancestors, including a skeleton that is almost entirely cartilaginous, and a paddle-shaped rostrum (snout) that extends nearly one-third their body length. They have been referred to as freshwater sharks because of their heterocercal tail or caudal fin, which resembles that of sharks. American paddlefish are a highly derived fish because they have evolved with adaptations such as filter feeding.  Their rostrum and cranium are covered with tens of thousands of sensory receptors for locating swarms of zooplankton, which is their primary food source.
American paddlefish are native to the Mississippi River basin and once moved freely under the relatively natural, unaltered conditions that existed prior to the early 1900s.   Paddlefish commonly inhabited large, free-flowing rivers, braided channels, backwaters, and oxbow lakes throughout the Mississippi River drainage basin, and adjacent Gulf drainages.  Their peripheral range extended into the Great Lakes, with occurrences in Lake Huron and Lake Helen in Canada until about 90 years ago.  American paddlefish populations have declined dramatically primarily because of overfishing, habitat destruction, and pollution.  Poaching has also been a contributing factor to their decline and will continue to be as long as the demand for caviar remains strong.  Naturally occurring American paddlefish populations have been extirpated from most of their peripheral range, as well as from New York, Maryland, Virginia, and Pennsylvania.  The current range of American paddlefish has been reduced to the Mississippi and Missouri River tributaries and Mobile Bay drainage basin. They are currently found in twenty-two states in the U.S., and those populations are protected under state, federal and international laws.'''

my_doc = spacy_tokenizer(sentence)
token_list = []
print(my_doc)
for token in my_doc:
    token_list.append(token.text)
print(token_list)
