# Trains a Word2Vec model using the captions of the InstaCites1M dataset

import gensim
import string
import glob
from stop_words import get_stop_words

# Output model path
model_path = '../datasets/InstaCities1M/models/word2vec_InstaCities1M.model'

# Word2Vec training config
size = 400 # word representation size
min_count = 5 # discard words with less than 5 appearances
epochs = 25 # iterate over the training corpus x times (train for x epochs)
window = 8 # words window used during training
training_cores = 8 # number of CPU cores used to train the model

# Finetuning usually is not very helpfull in text representations methods
finetune = False
if finetune:
    pretrained_model_path = '../datasets/InstaCities1M/models/word2vec_pretrained.model'
    model = gensim.models.Word2Vec.load(pretrained_model_path)

# Path of the dataset captions
text_data_path = '../datasets/InstaCities1M/captions_resized_1M/cities_instagram/'
cities = ['london','newyork','sydney','losangeles','chicago','melbourne','miami','toronto','singapore','sanfrancisco']

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


def get_instacities1m():
    # Loads image captions from InstaCites1M
    posts_text = []
    for city in cities:
        print("Loading InstaCities1M captions data from " + city)
        for i, file_name in enumerate(glob.glob(text_data_path + city + "/*.txt")):
            # if i == 100: break # Train a model with 1000 captions to test the code
            caption = ""
            filtered_caption = ""
            file = open(file_name, "r",  encoding="utf8")
            for line in file:
                caption = caption + line
            # Replace hashtags with spaces (sometimes people do no use spaces between hashtags)
            caption = caption.replace('#', ' ')
            # Keep only letters and numbers
            for char in caption:
                if char in whitelist:
                    filtered_caption += char
            posts_text.append(filtered_caption)
    return posts_text


print("Loading data ...")
posts_text = get_instacities1m()
print("Number of posts: " + str(len(posts_text)))

print("Creating tokens ...")
texts_2train = [] # List of lists of tokens to train the word2vec model, as Gensim Word2Vec requires


# Get over all captions, split them in words, remove stop words and prepare the data to train the word2vec model
c= 0
for t in posts_text:
    c += 1
    if c % 10000 == 0:
        print(c)
    # Handle exceptions just in case there is some corrupted data
    try:
        t = t.lower() # All to lowecase
        tokens = gensim.utils.simple_preprocess(t) # Split captions in words
        filtered_tokens = [i for i in tokens if not i in en_stop] # remove stop words from tokens
        texts_2train.append(filtered_tokens)
    except:
        continue

posts_text = [] # Clear unused data

#Train the model
print("Training ...")
if finetune:
    print("Finetunning the model from: " + str(model.iter))
    model.train(texts_2train, total_examples=model.corpus_count, epochs=epochs, compute_loss=False)
else:
    model = gensim.models.Word2Vec(texts_2train, size=size, min_count=min_count, workers=training_cores, iter=epochs, window=window)

model.save(model_path) # Save the model to disk
print("Training Completed")