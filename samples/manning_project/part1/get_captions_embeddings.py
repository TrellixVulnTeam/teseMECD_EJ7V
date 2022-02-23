# Computes the embeddings of the InstaCities1M dataset captions and saves them to disk
# The embedding of its caption is the mean of its words embeddings (normalized)
# It saves the embeddings in a txt file where each line contains the image ID and its caption embedding

from stop_words import get_stop_words
#from gensim import corpora, models
import glob
import string
import numpy as np
import gensim

# Load the data and the model
text_data_path = '../datasets/InstaCities1M/captions_resized_1M/cities_instagram/'
model_path = '../datasets/InstaCities1M/models/word2vec_InstaCities1M.model'
cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']

# # Create files to store the captions representations for each one of the splits
output_file_train = '../datasets/InstaCities1M/embeddings/train_InstaCities1M.txt'
output_file_val = '../datasets/InstaCities1M/embeddings/val_InstaCities1M.txt'
output_file_test = '../datasets/InstaCities1M/embeddings/test_InstaCities1M.txt'
train_file = open(output_file_train, "w")
val_file = open(output_file_val, "w")
test_file = open(output_file_test, "w")

# Load Word2Vec model
model = gensim.models.Word2Vec.load(model_path)
size = 400 # vector size
num_images_per_city = 100000
num_val = num_images_per_city * 0.05
num_test = num_images_per_city * 0.15

# We do the same text processing as the one we did to train the model (+ we will remove out of vocabulary words)
# We'll keep only letters and digits of the text (remove other symbols, emojis, etc)
whitelist = string.ascii_letters + string.digits + ' '
# Filter out some words from the data
# Here we filter out some common words that can harm the training because they are frequent and with general meaning
words2filter = ['http','https','photo','picture','image','insta','instagram','post']
# Filter out English stop words list ej ("and, "or", etc)
en_stop = get_stop_words('en')
# add own stop words
for w in words2filter:
    en_stop.append(w)


def compute_representation(file_name):

    img_id = file_name.split('/')[-1][:-4]

    with open(file_name, 'r', encoding="utf8") as file:

        caption = ""
        for line in file:
            caption = caption + line
        # Replace hashtags with spaces
        caption = caption.replace('#',' ')

        # Keep only letters and numbers
        filtered_caption = ""
        for char in caption:
            if char in whitelist:
                filtered_caption += char
        filtered_caption = filtered_caption.lower() # To lowercase
        tokens = gensim.utils.simple_preprocess(filtered_caption) # get words
        tokens_filtered = [i for i in tokens if not i in en_stop] # Filter out stop words
        tokens_filtered_in_vocab = [token for token in tokens_filtered if token in model.wv.vocab] # Filter out out of vocab tokens

        # Compute embedding of each word and the whole caption (average embedding of words)
        embedding = np.zeros(size)
        for tok in tokens_filtered_in_vocab:
            embedding += model[tok]

        # Normalize the embedding (div by number of words) 
        if len(tokens_filtered_in_vocab) > 1:
            embedding /= len(tokens_filtered_in_vocab)
        # (min at 0, # div by max)
        if min(embedding) < 0:
            embedding = embedding - min(embedding)
        if max(embedding) > 0:
             embedding = embedding / max(embedding)

        # Numpy to string to save
        out_string = ''
        for t in range(0,size):
            out_string = out_string + ',' + str(embedding[t])

        return city + '/' + img_id + out_string


for city in cities:
        print(city)
        
        captions_embeddings = []
        for file_name in glob.glob(text_data_path + city + "/*.txt"):
            embedding = compute_representation(file_name)
            captions_embeddings.append(embedding)
            # if len(captions_embeddings) == 100: break # Compute only embeddings of 100 captions to test code

        # Save the caption embeddings in a txt file, where each line contains an image ID and the embedding of its caption
        count = 0
        for em in captions_embeddings:
            # Create splits with the same number of images per city in each split
            if count < num_test:
                test_file.write(em + '\n')
            elif count < num_test + num_val:
                val_file.write(em + '\n')
            else:
                train_file.write(em + '\n')
            count += 1

# Close files
train_file.close()
val_file.close()
test_file.close()

print("Done")
