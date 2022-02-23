# Loads the precomputed image embeddings of the InstaCities1M test set
# Loads the Word2Vec model
# Computes distances between some textual queries (hardcoded here) and the image embeddings
# Finds the closest images to each query and saves the retrieval results to disk
# This code can handle simple single word queries and queries adding or substracting 2 words
# But queries could operate with more words, and queries can also be image embeddings, which can also be combined with word embeddings

import numpy as np
import operator
import os
from shutil import copyfile
import gensim 

dataset_root = '../../../datasets/InstaCities1M/'
img_embeddings_path = dataset_root + 'embeddings/test_img_embeddings.txt'
word2vec_model_path = dataset_root + '/models/word2vec/word2vec_model_InstaCities1M.model'
embedding_dimensionality = 400 
num_results = 20 # Num retrival results we want to save

# Load Word2Vec model
print("Loading Word2Vec model ...")
word2vec_model = gensim.models.Word2Vec.load(word2vec_model_path)

# Load image embeddings
img_embeddings = {}
file = open(img_embeddings_path, "r")
print("Loading img embeddings ...")
for line in file:
    d = line.split(',')
    regression_values = np.zeros(embedding_dimensionality)
    for t in range(0,embedding_dimensionality):
            regression_values[t] = d[t + 1]
    img_embeddings[d[0]] = regression_values

# Normalize the image embeddings (This is not needed but leads to better results)
for img_id in img_embeddings:
    img_embeddings[img_id] = img_embeddings[img_id] / sum(img_embeddings[img_id])

queries = ['car','nature','green','dog','food','sunset','london','man','skyline','night','beach','fashion','sydney', \
 ,'london+bridge','dog+park','dog+beach','beach+sunrise','sunrise-beach','hairstyle']


for q in queries:

    print("Computing results for query: " + q)

    # Handle complex queries that sum or substract two terms
    # TODO here we could also sum or substract image representations to use them as queries
    if '+' not in q and '-' not in q:
        q_embedding = word2vec_model[q]
    elif '+' in q:
        q_embedding = word2vec_model[q.split('+')[0]] + word2vec_model[q.split('+')[1]]
        q_embedding /= 2
    elif '-' in q:
        q_embedding = word2vec_model[q.split('-')[0]] - word2vec_model[q.split('-')[1]]
    else:
        print("Error in query")
        continue

    # Normalize query embedding the same way we did with during captions embeddings computation
    if min(q_embedding) < 0:
        q_embedding = q_embedding - min(q_embedding)
    if max(q_embedding) > 0:
        q_embedding = q_embedding / max(q_embedding)

    # Compute distances between query embedding and images embeddings
    similarities = {}
    for img_id, img_embedding in img_embeddings.items():
        similarities[img_id] = np.dot(img_embedding,q_embedding)
        # similarities[img_id] = np.linalg.norm(img_embedding-q_embedding)

    # Sort by ditance
    similarities = sorted(similarities.items(), key=operator.itemgetter(1), reverse=True)

    # Save results
    results_path = dataset_root + '/results/' + q + '/'
    if not os.path.exists(results_path):
        # print("Creating dir: " + results_path)
        os.makedirs(results_path)
    for idx, img_id in enumerate(similarities):
        print(img_id)
        result_name = img_id[0].replace('/','_')
        copyfile(dataset_root + '/img_resized_1M/cities_instagram/' + img_id[0] + '.jpg', results_path + result_name + '.jpg')
        if idx == num_results - 1: break

print("Done")