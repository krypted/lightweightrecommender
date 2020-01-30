# lightweightrecommender
A generic recommendation script built to run as a lambda or gcf

##Requirements
numpy
gensim
nltk==3.4.5
textblob==0.15.3


##Usage
Run locally, the recommender crawls through a column of a csv and matches the recommendations for similar content. 

Those are based on the content passed in the --text field. 

Can use the --recs option to define the number of recs you'd like to recieve in response. 

`python recommender.py --file='my.csv' --text="Flash update" --column="title" --recs=10`
