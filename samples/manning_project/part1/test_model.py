# Test the trained word2vec model exploring nearest words to some queries or words similarities
# We use some words which are typically used in Social Media so we can inspect the word representations we have learnt from it,
# and compare them which the ones learned by other standard Word2Vec models (ex, the typical one trained on Google News)
# https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html

from gensim import models
import gensim.downloader as api

model_google_news = api.load('word2vec-google-news-300') # This model is trained on a large dataset of news

model_name = 'word2vec_InstaCities1M.model'
model_path = '../datasets/InstaCities1M/models/' + model_name

print("Loading model ... ")
model = models.Word2Vec.load(model_path)

# Typical Word2Vec test example to check the semantic structure
# Find Most similar words
print("woman + king - man")
print(model.wv.most_similar(positive=['woman', 'king'], negative=['man']))
print("beach + sea - mountain")
print(model.wv.most_similar(positive=['beach', 'sea'], negative=['mountain']))

# Explore some Social Media related queries
print("toronto")
print(model.wv.most_similar(positive=['toronto']))
print("toronto + wild")
print(model.wv.most_similar(positive=['toronto', 'wild']))
print("london + water")
print(model.wv.most_similar(positive=['london', 'water']))
print("usa + water")
print(model.wv.most_similar(positive=['usa', 'water']))
print("vacation + usa")
print(model.wv.most_similar(positive=['vacation', 'usa']))
print("sports")
print(model.wv.most_similar(positive=['sports']))
print("hairstyle")
print(model.wv.most_similar(positive=['hairstyle']))
print("birthday + young")
print(model.wv.most_similar(positive=['birthday', 'young']))
print("family + happy")
print(model.wv.most_similar(positive=['happy', 'family']))
print("family - happy")
print(model.wv.most_similar(positive=['family'], negative=['happy']))
print("food")
print(model.wv.most_similar(positive=['food']))
print("food + healthy")
print(model.wv.most_similar(positive=['food', 'healthy']))
print("food + sweet")
print(model.wv.most_similar(positive=['food', 'sweet']))
print("food - healthy")
print(model.wv.most_similar(positive=['food'] , negative=['healthy']))

# Find the word that does not match
print("Outlier: breakfast cereal dinner lunch")
print(model.wv.doesnt_match(['breakfast','cereal','dinner','lunch']))
print("Outlier: man woman kid dog")
print(model.wv.doesnt_match(['man','woman','kid','dog']))

# Compute similarity between words
print("Similarity: woman', 'man")
print(model.wv.similarity('woman', 'man'))



