# Numeric Engineering 1(To be tested)
#For converting allnumeric data in columns like currency remperature, numbers in numeric form etc.. into numeric form
import pandas as pd
import numpy as np
import time
import holidays
import swifter
import spacy
from collections import Counter
from string import punctuation
from textblob import TextBlob
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
np.random.seed(2018)
import nltk
nltk.download('wordnet',quiet=True)
from gensim import corpora, models
stemmer = SnowballStemmer('english')

############################################
############## NUMERIC ENGINEERING #########
############################################

def numeric_engineering(df):
    start = time.time()

    def returnMoney(col):
        # Remove Commas from currencies
        try:
            return pd.to_numeric(col.str.replace([',','$','€','£','¥'],''))
        except:
            return col

    obj_columns= list(df.dtypes[df.dtypes == np.object].index)
    # print(f'object type columns are {obj_columns}')
    print(f'\t\t stripping spaces, symbols, and lower casing all entries')
    df[obj_columns]=df[obj_columns].apply(lambda x: x.astype(str).str.strip(' %$€£¥+').str.lower())
    print('done ...')
    print(f'\t\t Replacing empty and invalid strings')
    possible_empties = ['-','n/a','na','nan','nil',np.inf,-np.inf,'']
    df[obj_columns]=df[obj_columns].replace(possible_empties,np.nan)
    print('done ...')
    print(f'\t\t Replacing commas if present in Currencies')
    df[obj_columns]=df[obj_columns].apply(lambda x:returnMoney(x))
    print('done ...')
    obj_columns= list(df.dtypes[df.dtypes == np.object].index)
    df1 = df[obj_columns].copy()
    print(f'\t\t Finding Numeric Columns')
    df1 = df1.apply(lambda x : pd.to_numeric(x,errors='coerce'))
    df1.dropna(axis=1,thresh = 0.65*len(df),inplace=True)
    new_num_cols = df1.columns
    df[new_num_cols] = df[new_num_cols].apply(lambda x : pd.to_numeric(x,errors='coerce'))
    print('done ...')

    for i in df.columns :
        print(f'\t\t   {i} is of type {df[i].dtypes}')

    # # End of Testing codes
    end = time.time();print('Numeric Engineering time taken:',end - start);print('\n')
    return(df)


############################################
############## DATE ENGINEERING ############
############################################

def getDateColumns(df,withPossibilies=0):
    '''
    This method Identifies all columns with 'DATE' data by maximizing out the possibilities
    '''
    months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
    # First get all non-numerical Columns
    non_numeric_cols = df.select_dtypes('object')
    # This dictionary stores possibilities of a column containing 'DATES' based on the name of the column
    Possibility = {}
    for column in non_numeric_cols:
        if 'date' in column.lower():
            Possibility[column] = int(len(df)*0.1)
        else:
            Possibility[column] = 0
        for entry in df[column]:           # ITERATE THROUGH EVERY ENTRY AND TRY SPLITTING THE VALUE AND INCREMENT/DECREMENT POSSIBILITY
            try:                                                                      # USING EXCEPTION HANDLING
                if len(entry.split('/')) == 3 or len(entry.split('-')) == 3 or len(entry.split(':')) == 3:
                    Possibility[column] += 1
                    for month in months:
                        if month in entry.lower():
                            Possibility[column] += 1
                else:
                    Possibility[column] -= 1
            except:
                Possibility[column] -= 1
      # This contains the final DATE Columns
    DATE_COLUMNS = []
    for key,value in Possibility.items():
        if value > 0.8 * len(df):          # IF THE POSSIBILITY OF THE COLUMN IN GREATER THAN 1, THEN IT IS DEFINITELY A 'DATE COLUMN'
            DATE_COLUMNS.append(key)
            
    # to find missed date columns    
    def finddate(entry):    # function to find the presence of a month in an entry     
        a=0
        for month in months:
            if month in str(entry).lower():
                a=1
        return a
    
    Y=non_numeric_cols
    Y=Y.drop(DATE_COLUMNS, axis =1) 
    Possible_date_col=[]
    for col in Y.columns:
        a=Y[col].apply(finddate) # returns a series where value is one if the entry has a month in it
        if sum(a)>0.8*len(Y[col]):   # if there is a name of a month in atleast 80% entries of the column
            Possible_date_col.append(col)
            
    if not withPossibilies:
        return DATE_COLUMNS,Possible_date_col
    else:
        return DATE_COLUMNS,Possible_date_col,Possibility

def date_engineering(df):
    import itertools
    
    def fixdate(entry):    # function to introduce '-' before and after month and and removing timestamp if it is seperated from date by':' 
        months = ['jan','feb','mar','apr','may','jun','jul','aug','sep','oct','nov','dec']
        for month in months:
            if month in str(entry).lower():
                index1= entry.find(month)
                index2= index1+3
                entry = entry[:index1]+'-'+entry[index1:index2]+'-'+entry[index2:]
                index3=entry.find(':')  #only specific to messy dataset
                entry=entry[:index3]
        return entry  
    
    start = time.time()
    
    if possible_datecols:
        for col in possible_datecols:
            df[col]=df[col].apply(fixdate)
            
    print('\n\t Entering Date Engineering')
    df = df.apply(pd.to_datetime,errors='coerce')
    print('\nPrinting Missing % of date columns')
    MISSING = pd.DataFrame(((df.isnull().sum().sort_values(ascending=False)*100/len(df)).round(2)),columns=['Missing in %'])[:10]
    print(MISSING)
    
    if validation==False:   # dropping columns with missing greater than 35% only in training not scoring
        print('Dropping Columns with missing greater than 35% of total number of entries')
        df.dropna(thresh=len(df)*0.65,axis=1,inplace=True)
    try:     
        for c in df.columns:
            df[c].fillna(df[c].mode()[0],inplace=True)    # replacing null values with mode 

    except:
        for c in df.columns:
            df[c].fillna(df[c].mean(),inplace=True)   # if error in mode then replacing null values with mean      
     
    date_cols = df.columns
    visualize_dict = dict.fromkeys(date_cols, [])

    # creating separate month and year columns, and difference from current date
    for i in date_cols:
        df[str(i)+"_month"] = df[str(i)].dt.month.astype(int)
        df[str(i)+"_year"] = df[str(i)].dt.year.astype(int)
        df[str(i)+"-today"] = (pd.to_datetime('today')-df[str(i)]).dt.days.astype(int)
        visualize_dict[str(i)] =  visualize_dict[str(i)] + [str(i)+"_month"] + [str(i)+"_year"]+[str(i)+"-today"]

    # create difference columns
    diff_days = list()
    if (len(date_cols)>1) :
        for i in itertools.combinations(date_cols,2):
            diff_days = diff_days + [str(i[0])+"-"+str(i[1])]
            df[str(i[0])+"-"+str(i[1])]=(df[i[0]]-df[i[1]]).dt.days.astype(int)


    print('\n\t #### RUNNING WAIT ####')

    # See Near Holiday or not
    def nearHol(currentDate, us_hols, currentYear):
        new_list = []
        append = new_list.append
        for date, occasion in us_hols:
            if(date.year == currentYear):
                append(date)
        flag = 1
        for i in new_list:
            a = (currentDate.date()-i).days

            if abs(a)<=5:flag =1;break
            else:flag = 0

        return 0 if flag == 0 else 1

    for col in date_cols:
#         print('LOOP')
        #creating a unique list of all years corresponding to a column to minimise mapping
        us_hols = holidays.US(years=df[str(col)+'_year'].unique(), expand= False)
        #creating a new columns to check if a date falls near a holiday
        df[str(col)+'_Holiday'] = df.apply(lambda x: nearHol(x[col],us_hols.items(),x[str(col)+'_year']),axis=1)
        visualize_dict[str(col)] =  visualize_dict[str(col)] + [str(col)+"_nearestHoliday"]

    print("\nVisualizing Coloumns Generated\n {}" .format(visualize_dict))
    print("\nThe Following columns were generated to get days between dates of two seperate date columns\n {}".format(diff_days))
    end = time.time()
    print('\nDate Engineering Time Taken : {}'.format(end-start))
    print('\n\t #### DONE ####')
    return df.drop(date_cols,axis=1)


############################################
############## TEXT ANALYTICS ##############
############################################

def findReviewColumns(df): #input main dataframe

  rf = df.sample(n=150, random_state=1).dropna(axis=0) if len(df)>150 else df.dropna(axis=0)#use frac=0.25 to get 25% of the data

  #df.dropna(axis=0,inplace=True) #dropping all rows with null values



  #categorical_variables = []
  col_list =[]
  for col in rf.columns:
    if df[col].nunique() <100:
      col_list.append(col)           #define threshold for removing unique values #replace with variable threshold
      rf.drop(col, axis=1,inplace=True) #here df contains object columns, no null rows, no string-categorical,


  rf.reset_index(drop=True,inplace=True)
  for col in rf.columns:
        count1,count2,count3,count4 = 0,0,0,0
        for i in range(len(rf)):
            val = len(str(rf.at[i,col]).split())
            if val == 1:
                count1 = count1+1
            elif val == 2:
                count2 = count2+1
            elif val == 3:
                count3 = count3+1
            elif val == 4:
                count4 = count4+1
        print(col,"count of words is",count1,"-",count2,"-",count3,"-",count4,"-")

        if count1+count2+count3+count4 >=0.75*len(rf):
            col_list.append(col)
            print("dropping column",col)
            rf.drop(col, axis=1,inplace=True)





  start = time.time()
  print(rf.shape)
  nlp = spacy.load('en_core_web_sm', disable=['tagger','parser','textcat'])
  sf = pd.DataFrame()
  for col in rf.columns:
    sf[col] = rf[col].apply(nlp)


  end = time.time()
  print("Time taken to tokenize the DataFrame",end - start)

  #print("Tokenised Sampled DataFrame",sf)
  #print("Sampled DataFrame",rf)
  #print("Actual Dataframe",df)

  start = time.time()
  #testf = sf.sample(frac=0.10,random_state=44)

  #code to eliminate columns of name, city, address
  for col in sf.columns:
    entity_list =[]
    tokens = nlp(''.join(str(sf[col].tolist()))) #converting one column into tokens
    #print("the tokens of each column are:", tokens)
    token_len = sum(1 for x in tokens.ents)
    print("Length of token entities",token_len)                                    #create two lists that hold the value of actual token entities and matched token entities respectively
    if token_len>0:
      for ent in tokens.ents:
        if (ent.label_ == 'GPE') or (ent.label_ =='PERSON'):  #matching is done on the basis of whether the entity label is
          entity_list.append(ent.text)          #countries, cities, state, person (includes fictional), nationalities, religious groups, buildings, airports, highways, bridges, companies, agencies, institutes, DATE etc.

      entity_counter = Counter(entity_list).elements()  #counts the match
      counter_length = sum(1 for x in entity_counter)
      print("Length of matched entities",counter_length) #if there is at least a 50% match, we drop that column TLDR works better on large corpus
      if (counter_length >= 0.60*token_len):
        col_list.append(col)
    else:
      print("Length of token entities 0")
      print("Length of matched entities 0")
    counter_length = 0
    token_len = 0


  print("Columns that are going to be removed are ", col_list)   #list of columns that need to be removed
  ##########IMPORTANT LINE NEXT###############
  rf = df.copy() #unhide this to immediately work with the entire dataset and not just sampled dataset and vice-versa to work with sampled
  ##########DO NOT IGNORE ABOVE LINE##########
  for val in col_list:
    rf.drop(val, axis=1, inplace=True)
  end = time.time()
  print("Time taken for completion of excess column removal:", end-start)

  if (len(rf.columns) ==0):
    print("No Remarks or Comments Found ")
    flag = 0
    return None, None
  else:
    flag = 1

  if (flag == 1):
    main_list = [] #holds all the review columns
    append = main_list.append
    for col in rf.columns:
      append(col)

    return main_list, col_list

def sentiment_analysis(rf):
  bf = pd.DataFrame()
  def getSubjectivity(text):
    try:
        return TextBlob(text).sentiment.subjectivity #returns subjectivity of the text
    except:
        return None

  def getPolarity(text):
    try:
        return TextBlob(text).sentiment.polarity  #returns polarity of the sentiment
    except:
        return None

  for col in rf.columns:      #creating a new DataFrame with new columns
    col_pname = "{}-{}".format(col,"Polarity")
    col_sname = "{}-{}".format(col,"Subjectivity")
    bf[col_pname] = rf[col].apply(getPolarity)
    bf[col_sname] = rf[col].apply(getSubjectivity)

  return bf

def lemmatize_stemming(text):
    return stemmer.stem(WordNetLemmatizer().lemmatize(text, pos='v')) #performs lemmatization

def preprocess(text):
    result = []
    for token in gensim.utils.simple_preprocess(text):
        if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) > 3:  #removes stopwords and tokens with len>3
            result.append(lemmatize_stemming(token))
    return result

def topicExtraction(df,validation=False,lda_model_tfidf=None):

  data_text = df.copy()
  data_text['index'] = data_text.index
  documents = data_text

  headline = list(documents.columns)[0] #review column

  processed_docs = documents[headline].map(preprocess) #preprocessing review column

  #print("Processed Docs are as follows",processed_docs[:10])

  dictionary = gensim.corpora.Dictionary(processed_docs) #converting into gensim dict
  dictionary.filter_extremes(no_below=10,no_above=0.25, keep_n=1000)   #taking most frequent tokens

  bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs] #document to bag of words

  if validation==False:
    #print("BOW Corpus", bow_corpus[:10])
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus] #generating the TF-IDF of the corpus

    start = time.time()
    lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=10, id2word=dictionary, passes=1, workers=6) #multiprocessing Latent Dirilichtion Allocation Model
    end = time.time()
    print(end-start)
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic)) #printing topics in the corpus


  ser = []
  append = ser.append
  print("Bag of Words Corpus length",len(bow_corpus))
  start = time.time()
  for i in range(len(bow_corpus)):
    for idx, topic in sorted(lda_model_tfidf[bow_corpus[i]], key= lambda tup: -1*tup[1]):
      append(idx)
      break
  end = time.time()
  asf = pd.DataFrame(ser)
  print("Time for append", end-start)

  return asf, lda_model_tfidf
