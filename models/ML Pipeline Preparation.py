#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[14]:


# import libraries
import sys
import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from sqlalchemy import create_engine

import pickle


# In[37]:


# load data from database
engine = create_engine('sqlite:///Disaster_messages.db')
df = pd.read_sql_table('Disaster_messages', engine)
X = df['message']
Y = df.drop(['id','message','original', 'genre','is_dupe'], axis=1)


# In[33]:


np.sum(df.iloc[:,4:],axis=1).tolist()


# In[4]:


categories=list(Y)


# In[5]:


Y.info()


# ### 2. Write a tokenization function to process your text data

# In[6]:


def tokenize(text):
    tokens=word_tokenize(text)
    Lemmatizer=WordNetLemmatizer()
    tokens_lem=[Lemmatizer.lemmatize(t).lower().strip() for t in tokens]
    return tokens_lem


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[7]:


def model_pipeline():
    pipeline =Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))   
        ])
    return pipeline


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[7]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=101)


# In[9]:


model=model_pipeline()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)


# In[10]:


y_pred


# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[11]:


accuracy = (y_pred == Y_test).mean()
accuracy


# In[12]:


for i in range(36):
    print("Category:", categories[i],"\n", classification_report(Y_test.values[:, i], y_pred[:, i]))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[13]:


def build_model():
    '''
    Finds the best parameters for RF model
    param: model
    param: X_train
    param: Y_train
    return: best set of parameters
    '''
    pipeline =Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))   
        ])
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 150],
        'clf__estimator__max_features': ['sqrt',],
        'clf__estimator__criterion': ['entropy', 'gini']
    }

    cv = GridSearchCV(pipeline, param_grid = parameters, verbose=1)

    return cv


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[14]:


model=build_model()
model.fit(X_train, Y_train)
y_pred = model.predict(X_test)


# In[16]:


model.best_params_


# In[17]:


accuracy = (y_pred == Y_test).mean()
accuracy


# In[18]:


for i in range(36):
    print("Category:", categories[i],"\n", classification_report(Y_test.values[:, i], y_pred[:, i]))
#we see improvements in a few categories like request, aid_related, food weather related and direct report


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# In[9]:


class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)


# In[8]:


def new_model():
    '''
    Finds the best parameters for RF model
    param: model
    param: X_train
    param: Y_train
    return: best set of parameters
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(criterion='gini',max_features='sqrt',n_estimators=100)))   
        ])
    parameters = {
        #'vect__ngram_range': ((1, 1), (1, 2)),
         'vect__max_df': (0.75, 1.0)
    }

    cv = GridSearchCV(pipeline, param_grid = parameters, verbose=1,n_jobs=-1)

    return cv


# In[9]:


newmodel=new_model()
newmodel.fit(X_train, Y_train)
y_pred = newmodel.predict(X_test)


# ### 9. Export your model as a pickle file

# In[10]:


with open('newmodel.pkl', 'wb') as file:
    pickle.dump(newmodel, file)


# In[ ]:





# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[38]:


# import libraries
import sys
import pandas as pd
import numpy as np
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier,GradientBoostingClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_recall_fscore_support

from sqlalchemy import create_engine

import pickle

def load_data(filepath):
    '''
    Load data function
    Inputs:
    filepath: path to SQLite db
    Outputs:
    X:features 
    Y:label
    category_names: types of disaster
    '''
    engine = create_engine('sqlite:///{}'.format(filepath))
    df = pd.read_sql_table('Disaster_messages', engine)
    X = df['message']
    Y = df.drop(['id','message','original', 'genre','is_dupe'], axis=1)
    category_names=Y.columns
    return X,Y,category_names

def tokenize(text):
    '''
    inputs: 
    text: list of text messages
    outputs:
    tokens_lem: tokenized text
    '''
    tokens=word_tokenize(text)
    Lemmatizer=WordNetLemmatizer()
    tokens_lem=[Lemmatizer.lemmatize(t).lower().strip() for t in tokens]
    return tokens_lem

class StartingVerbExtractor(BaseEstimator, TransformerMixin):

    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
def build_model():
    '''
    Finds the best parameters for RF model
    param: model
    param: X_train
    param: Y_train
    return: best set of parameters
    '''
    pipeline =Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))   
        ])
    parameters = {
        'clf__estimator__n_estimators': [50, 100, 150],
        'clf__estimator__max_features': ['sqrt',],
        'clf__estimator__criterion': ['entropy', 'gini']
    }

    cv = GridSearchCV(pipeline, param_grid = parameters, verbose=1)

    return cv

def evaulate_model(model,X_test,Y_test,category_names):
    '''
    This function will evaluate the model performance
    Input:
    X_test:test features
    y_test:test target
    category_names: labels names
    '''
    y_pred = model.predict(X_test)
    for i in range(36):
        print("Category:", categories[i],"\n", classification_report(Y_test.values[:, i], y_pred[:, i]))
        
def save_model(model,model_path):
    '''
    This function saves the model in a pkl file
    Input:
    model
    model_path
    Output: 
    pickle file of the saved model
    '''
    with open('newmodel.pkl', 'wb') as file:
        pickle.dump(newmodel, file)
        
def main():
        if len(sys.argv) == 3:
            database_filepath, model_filepath = sys.argv[1:]
            print('Loading data...\n    DATABASE: {}'.format(database_filepath))

            X, Y, category_names = load_data(database_filepath)
            X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2,random_state=101)
            print('Building model...')
            model = build_model()
            print('Training model...')
            model.fit(X_train, Y_train)
            print('Evaluating model...')
            evaluate_model(model, X_test, Y_test, category_names)
            print('Saving model...\n    MODEL: {}'.format(model_filepath))
            save_model(model, model_filepath)
            print('Trained model saved!')
        else:
            print('Please provide the filepath of the disaster messages database '              'as the first argument and the filepath of the pickle file to '              'save the model to as the second argument. \n\nExample: python '              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


        
if __name__ == '__main__':
    main()


# In[ ]:




