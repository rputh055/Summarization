from random import sample

import csv
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
import json 
from transformers import T5Tokenizer, T5ForConditionalGeneration, T5Config

class Clustering:
    def __init__(self, path):
        with open(path + '/two_processed.csv') as csvfile:
            readCSV = csv.reader(csvfile ,delimiter = ',')
            a = []
            for row in readCSV:
                tweet = row[0]
                a.append(tweet)

            self.sampled_tweets = sample(a,1000)

            model = SentenceTransformer('bert-base-nli-mean-tokens')

            self.tweet_embeddings = model.encode(self.sampled_tweets)

    def elbow(self, path):

        distortions = []
        K = range(1,10)
        for k in K:
            kmeanModel = KMeans(n_clusters=k)
            kmeanModel.fit(self.tweet_embeddings)
            distortions.append(sum(np.min(cdist(self.tweet_embeddings, kmeanModel.cluster_centers_, 'euclidean'), axis=1)) / self.tweet_embeddings[1].shape[0])

        plt.plot(K, distortions, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Distortion')
        plt.title('The Elbow Method showing the optimal k')
        #plt.style.use('dark_background')
        plt.savefig(path +'/plot')
        

    def cluster(self, path, k_value):
        
        kmeans = KMeans(n_clusters=k_value)
        kmeans.fit(self.tweet_embeddings)
        y_kmeans = kmeans.predict(self.tweet_embeddings)

        clusters_df = pd.DataFrame(
        {"text" : self.sampled_tweets,
        "cluster" : y_kmeans})
        for i in range(k_value):

            df = clusters_df[clusters_df['cluster']==i]

            df.drop(columns=['cluster'],inplace=True)

            df.to_csv(path +'/tweets'+str(i)+'.txt')

  
        

            with open(path +'/tweets'+str(i)+'.txt', 'r') as file:
                data = file.read().replace('\n', '')

            model = T5ForConditionalGeneration.from_pretrained('t5-small')
            tokenizer = T5Tokenizer.from_pretrained('t5-small')
            device = torch.device('cpu')

            preprocess_text = data.strip().replace("\n","")

            t5_prepared_Text = "summarize: "+preprocess_text
            #print ("original text preprocessed: \n", preprocess_text)

            tokenized_text = tokenizer.encode(t5_prepared_Text, return_tensors="pt").to(device)

            # summmarize 
            summary_ids = model.generate(tokenized_text,
                                                num_beams=4,
                                                no_repeat_ngram_size=2,
                                                min_length=30,
                                                max_length=100,
                                                early_stopping=True)

            output = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

            #print ("\n\nSummarized text: \n",output)
            with open(path + '/summary'+str(i)+'.txt', 'w') as sum:
                sum.write(output)

