# 1. Library imports
import uvicorn
from fastapi import FastAPI
import tensorflow as tf
from spacy.tokens import Token
from spacy.tokens import Doc
from spacy.lang.en.stop_words import STOP_WORDS
from spacy.lang.en import English
from nltk.stem.snowball import SnowballStemmer
from spacy.language import Language
import numpy as np
import spacy
#import numpy as np
from tensorflow.keras.models import load_model
from transformers import BertTokenizer, TFBertModel, BertConfig, TFBertForSequenceClassification
from spacymoji import Emoji
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 2. Create app and model objects
app = FastAPI()
#model = tf.keras.models.load_model('output/models/bert_model.h5')

#emoji = Emoji(nlp)
#_ = nlp.add_pipe("emoji", first=True)
#print("info", nlp.pipe_names)

def is_mention_function(token):
    if token.text.startswith("@") == True and len(token.text) > 1:
        return True

nlp = spacy.load("en_core_web_sm")
#emoji = Emoji(nlp)
_ = nlp.add_pipe("emoji", first=True)
nlp.Defaults.stop_words.add("re")
nlp.Defaults.stop_words -= {"not", "no", "n't", "would", "without", 
                            "could", "still", "ever",
                            "yet", "almost", "should", 
                            "always", "too", "sometimes", "except",
                            "everything", "really", "nothing", 
                            "down", "also", "very", "most", "'d",
                            "serious", "than", "however", "well",
                            "neither", "anyhow", "few",
                            "rather", "mostly", "none", "must", 
                            "less", "many", "as", "often",
                            "never", "enough", "much", "out", "but", 
                            "whereas", "netherrless",
                            "next", "even", "although", "why", 
                            "again", "perhaps"}
Token.set_extension("is_mention", getter=is_mention_function, force=True) 


# load model
#model = tf.keras.models.load_model('./output/models/model_bert_final.h5')
#model = load_model('./output/models/model_bert_final.h5')


bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")





# Définit un composant personnalisé
def text_clean_function(doc):
    new_words = []
    for token in doc:
        #if token._.is_emoji:
            #if token._.emoji_desc != " ":
                #new_words.append(token._.emoji_desc)
            #else:
                #continue
        if token.like_email or token.is_stop or token.like_url:
            continue
        elif token._.is_mention:
            continue
        elif token.is_punct:
            continue
        else:
            if token.text.strip() != "":
                new_words.append(token.text)
    return new_words


#@Language.component("pre_process_text")
def pre_process_text_function(doc):
    # Generate a new list of tokens here
    new_words = text_clean_function(doc)
    new_doc = Doc(doc.vocab, words=new_words)
    return new_doc


def text_cleaning(text, remove_stop_words=True, lemmatize_words=True):
    # nlp.add_pipe("pre_process_text", before="tok2vec")
    doc = nlp(text)
    new_words = text_clean_function(doc)
    new_doc = Doc(doc.vocab, words=new_words)
    result = [token.text for token in new_doc]
    result_2 = " ".join(result)
    return result_2

@app.get("/")
async def root():
    word=text_cleaning("im sad")
    return {"message": word}
#"Welcome to the Feelings API!"+

# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted flower species with the confidence
@app.get("/predict-review")
def predict_sentiment(review: str):
    """
    A simple function that receive a review content
    and predict the sentiment of the content.
    :param review:
    :return: prediction, probabilities
    """
    # clean the review
    cleaned_review = text_cleaning(review)
    # perform prediction
    input_ids=[]
    attention_masks=[]
    bert_inp=bert_tokenizer.encode_plus(cleaned_review,add_special_tokens = True,max_length=64,pad_to_max_length = True,return_attention_mask = True)
    input_ids = bert_inp['input_ids']
    attention_masks =bert_inp['attention_mask']


    model_save_path='./output/models/bert_model.h5'
    
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    metric = tf.keras.metrics.SparseCategoricalAccuracy('auc')
    optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5,epsilon=1e-08)

    trained_model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased',num_labels=2)
    trained_model.compile(loss=loss,optimizer=optimizer, metrics=[metric])
    trained_model.load_weights(model_save_path)

    preds = trained_model.predict([input_ids,attention_masks],batch_size=32)
    pred_labels = np.argmax(preds.logits, axis=1)
    #preds = model.predict("test",batch_size=32)
    

    #preds = model.predict([val_inp,val_mask],batch_size=32)
    #print("--------------------", preds)
    #pred_labels = np.argmax(preds.logits, axis=1)
    #prediction = model.predict([cleaned_review])
    #y_prob = np.max(preds)#int(pred_labels[0])
    #y_pred = np.where(y_prob > 0.5, 1, 0)
    #probas = model.predict_proba([cleaned_review])
    # output dictionary
    sentiments = ["Negative", "Positive"]
    # show results
    result = {"prediction": sentiments[int(pred_labels[0])]}
    return result


# 4. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)