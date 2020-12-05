# Numeric Engineering 1(To be tested)
#For converting allnumeric data in columns like currency remperature, numbers in numeric form etc.. into numeric form
import pandas as pd
import numpy as np
import time
import itertools
from math import sin,cos,sqrt,pow
import decimal
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
from wordcloud import WordCloud,STOPWORDS 
from IPython.display import Image 
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

def date_engineering(df, possible_datecols, validation=False):
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
  
  print("Names of columns are",df.columns)
  data_text = df.copy()
  data_text['index'] = data_text.index
  documents = data_text

  headline = list(documents.columns)[0] #review column

  processed_docs = documents[headline].map(preprocess) #preprocessing review column

  #print("Processed Docs are as follows",processed_docs[:10])

  dictionary = gensim.corpora.Dictionary(processed_docs) #converting into gensim dict
  dictionary.filter_extremes(no_below=10,no_above=0.25, keep_n=1000)   #taking most frequent tokens
  
  if validation==False:
    print("Generating Wordcloud...")
    word_str = ""
    for k,v in dictionary.token2id.items():
        word_str = word_str + k + " "
    # Generating wordcloud
    wordcloud = WordCloud(width = 1000, height = 800, random_state=42, background_color='white', colormap='twilight', collocations=False, stopwords = STOPWORDS).generate(word_str)
    #Saving the image
    file = str(df.columns[0])+"_wordcloud.png"
    wordcloud.to_file(file)
    image = Image(filename=file)
    display(image)

  bow_corpus = [dictionary.doc2bow(doc) for doc in processed_docs] #document to bag of words

  if validation==False:
    #print("BOW Corpus", bow_corpus[:10])
    tfidf = models.TfidfModel(bow_corpus)
    corpus_tfidf = tfidf[bow_corpus] #generating the TF-IDF of the corpus
    print("!!!!!!",len(corpus_tfidf))
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
  count = 0
  for i in range(len(bow_corpus)):
    val = bow_corpus[i]
    # for i in range(len(val)):
    #     print("Word {} (\"{}\") appears {} time.".format(val[i][0], dictionary[val[i][0]], val[i][1]))
    # count = count+1
    for idx, topic in sorted(lda_model_tfidf[bow_corpus[i]], key= lambda tup: -1*tup[1]):
      append(idx)
      break
    # print("Loop ran for ",count)
  end = time.time()
  asf = pd.DataFrame(ser)
  print("Time for append", end-start)

  return asf, lda_model_tfidf



############################################
############## LAT-LONG ENGINEERING ##############
############################################

def floatCheck(x):
    if isinstance(x,float) is True:# Accepts floating point numbers as well as np.nan  (testing)#and float.is_integer(x) is False:
        return True
    else:
        return False

def checkFormat(x):
    try:
        if (decimal.Decimal(str(x)).as_tuple().exponent <= -3) and (x>-180.0 and x<180.0): 
            return True
        else:
            return False
    except:
        return False

def checkLatLong(x):
    if x > -180.0 and x < 180.0:
        if x > -90.0 and x < 90.0:
            return "Lat"
        return "Long"
    else:
        return np.nan

def checkCondition(x):
    try:
        if x[0]=='(' and x[-1]==')' and len(x.split(','))==2:
            return True
        else:
            return False
    except TypeError:
        return np.nan

def Floater(df,value):
    print(value)
    floaters = []
    for column in df.columns: #add try catch block
        # print(f"testing column {column}")
        if value == "returnFloat":
            a = df[column].apply(lambda x: floatCheck(x)).to_list()
        elif value == "confirmLatLong":
            a = df[column].apply(lambda x: checkFormat(x)).to_list()
        # print(f"printing true value counts {a.count(True)}")
        if a.count(True) >0.9*len(df):
            floaters.append(column)
    return floaters

def segregator(df):
    lat_cols = []
    long_cols = []
    for col in df.columns:
        a = df[col].apply(lambda x: checkLatLong(x)).to_list()
        if a.count("Lat") > 0.9*len(df):
            lat_cols.append(col)
        elif a.count("Long") > 0.9*len(df):
            long_cols.append(col)
    if not long_cols:
        try:
            for i in range(1,len(lat_cols),2):
                long_cols.append(lat_cols[i])
                lat_cols.remove(lat_cols[i])
        except:
            print("Lat-Long Length Mismatch")
            lat_cols = []
            long_cols = []
    return lat_cols,long_cols

def pseudoFormat(df):
    lat_long_cols = []
    for col in df.columns:
        a = df[col].apply(lambda x: checkCondition(x)).to_list()
        if a.count(True) > 0.9*len(df):
            lat_long_cols.append(col)
    return lat_long_cols
def findLatLong(df):
    lat_cols = []
    long_cols = []
    lat_long_cols = [] # will add logic for lat-long columns of the form (lat,long)
    lat_long_cols = pseudoFormat(df)
    if lat_long_cols:
        print("Columns that are are of the form Lat-Long are as follows",lat_long_cols)
    columns = Floater(df,"returnFloat")  #List of float columns that could be lat or long
    print(f"The columns that could be Lat/Long are as follows {columns}")
    desired = [] 
    requisites = ["Lat","Long","Latitude","Longitude"]
    for val in itertools.product(columns,requisites):
        if val[0].lower().find(val[1].lower()) != -1 and val[0] not in desired:
            desired.append(val[0])
            columns.remove(val[0])
    #Removing columns with low nunique()
    for col in columns[:]: 
        if df[col].nunique() <100:
            columns.remove(col)
    #We check if there are any lat or long columns present in the rest of the float columns
    if columns:
        possible = Floater(df[columns],"confirmLatLong")
        if possible: #If they are of Lat Long format then add it to desired list
            desired.extend(possible)

    if desired:
        lat_cols, long_cols = segregator(df[desired])


    
    return lat_cols, long_cols,lat_long_cols

def distanceCalc(x_list):
    x = cos(float(x_list[0])) * cos(float(x_list[1]))
    y = cos(float(x_list[0])) * sin(float(x_list[1]))
    z = sin(float(x_list[0]))
    return sqrt(pow(x-1,2)+pow(y-0,2)+pow(z-0,2))

def convertCartesian(x):
    try:
        x_list = x[x.find('(')+1:x.find(')')].split(',')
        return distanceCalc(x_list)
    except AttributeError:
        return 0.0
def originGenerator(latitude,longitude):
    temp = {}
    temp[latitude.name] = latitude
    temp[longitude.name] = longitude
    temp_df = pd.DataFrame.from_dict(temp)
    ser = temp_df.apply(lambda x:distanceCalc([x[0],x[1]]),axis=1)
    return ser 
def latlongEngineering(df,lat_cols,long_cols,lat_long_cols):
    req = {}
    if lat_long_cols:
        for c in lat_long_cols:
                ser = df[c].apply(lambda x: convertCartesian(x))
                req[f'{c}-Origin'] = ser 
    if lat_cols and long_cols and len(lat_cols) == len(long_cols):
        for i in range(len(lat_cols)):  
            ser = originGenerator(df[lat_cols[i]],df[long_cols[i]])
            ser.fillna(0.0,inplace=True)
            req[f'{lat_cols[i]}_{long_cols[i]}-Origin'] = ser
    else:
        print("Lat columns and Long columns length mismatch")

    if req:
        return pd.DataFrame.from_dict(req) 
    else:
        return None

############################################
############## LAT-LONG ENGINEERING ##############
############################################

############################################
############## EMAIL URL ENGINEERING ##############
############################################

################### EMAIL AND URL IDENTIFICATION FUNCTIONS ###################
def identifyEmailUrlColumns(df,email=True): # Default email parameter is true for email identification
                                            # Email parameter is false for URL identification
    # Identification of columns having email addresses
    start = time.time()
    
    
    # Creating a dictionary with column names as keys and possibilities of that column
    # being an email/url as values    
    possibilities = {}
    
    if email:
        print('\nIdentifying Email Columns\n')
        # Initializing possibilities dictionary 
        for column in df:
            if ('mail' in column):          # If the string 'mail' is found in column names, initialize with 10% possibility
                possibilities[column] = 0.1
            else:                           # Else initialize with 0 possibility
                possibilities[column] = 0            
                
    else:
        print('\nIdentifying URL Columns\n')
        # Initializing possibilities dictionary 
        for column in df:
            if ('url' in column) or ('link' in column): # If the string 'url or link' is found in column names
                possibilities[column] = 0.1
            else:                           # Else initialize with 0 possibility
                possibilities[column] = 0     
        
    if email:
        # For every column in df, for every entry in a column, if the entry has @ and . in it, increase possibility 
        # by a fraction calculated by the length of the dataframe
        for column in df:
            for entry in df[column]:
                if ('@' in entry) and ('.' in entry) and (' ' not in entry):
                    possibilities[column] += 100/len(df)
                elif entry == '': # If we found empty string, do nothing
                    pass
                else:
                    possibilities[column] -= 100/len(df)      
    else:
        # For every column in df, for every entry in column, if the entry has https:// or http:// in it, increase possibility 
        # by a fraction calculated by the length of the dataframe
        for column in df:
            for entry in df[column]:
                if (('https://' in entry) or ('http://' in entry)) and (entry.count('.')>0):
                    possibilities[column] += 100/len(df)
                elif entry == '': # If we found empty string, do nothing
                    pass
                else:
                    possibilities[column] -= 100/len(df)
                    
    # Converting dictionary to series, the possibility series - poss_series               
    poss_series = pd.Series(possibilities)
    
    if email:
        print('The possibilities of columns being email_address are as below:\n')
        print(poss_series)
        # If possibility of a columns is greater than -45, then we consider it
        email_cols = poss_series[poss_series>-45].index    
        print('\nThe email columns found are: {}'.format(email_cols)) 
        end = time.time()
        print('Email Address identification time taken : {}'.format(end-start))
        return email_cols
    else:
        print('The possibilities of columns being url are as below:\n')
        print(poss_series)
        # If possibility of a columns is greater than -45, then we consider it
        url_cols = poss_series[poss_series>-45].index    
        print('\nThe url columns found are: {}'.format(url_cols)) 
        end = time.time()
        print('URL identification time taken : {}'.format(end-start))
        return url_cols


################### EMAIL AND URL ENGINEERINGS ###################
def emailUrlEngineering(df,email=True): # Default email parameter is true for email engineering
                                        # Email parameter is false for URL engineering
    ############################## EMAIL ENGINEERING ##############################
    
    start = time.time()
    
    if email is True:
        print('\n########## EMAIL ENGINEERING RUNNING ##########')
    else:
        print('\n########## URL ENGINEERING RUNNING ##########')

    def getEmailDomainName(col):
            # Get the first domain name, example: a@b.gov.edu.com, in this b alone is taken
            try:
                # print("Inside email")
                # print(col[1].split('.')[0])
                return col[1].split('.')[0]
            except:
                return np.nan # Invalid Entry

    def getUrlDomainName(col):
            try:
                # print("Inside url")
                # print(col.split('://')[1].split('/')[0].split('.')[0])
                return col.split('://')[1].split('/')[0].split('.')[0]
            except:
                return np.nan # Invalid Entry

    
    # Making a note of newly created columns
    newCols = []
    # For every column in email columns, get Domain Name, create a new column and check the missing percentage
    for column in df.columns:
        domain_name = column + '_domain'
        if email is True:
            df[domain_name] = df[column].str.rsplit('@')
            df[domain_name] = df[domain_name].apply(getEmailDomainName)
        else:
            df[domain_name] = df[column].apply(getUrlDomainName)
        # Checking percentage of missing values
        if df[domain_name].isnull().sum()/len(df) >= 0.5:
            print('The newly created \'{}\' column has 50% or more missing values!'.format(domain_name))
            print('And hence will be dropped!')
            df.drop(domain_name,axis=1)
        else:
            newCols.append(domain_name)
            
    if len(newCols) == 0:
        return_df = pd.DataFrame(None) # Will help return empty dataframe
    else:
        return_df = pd.DataFrame(df[newCols].fillna('missing')) # Returning DataFrame that contain only newly created columns
                                                                # With missing imputation done
        
    end = time.time()
    if email is True:
        print('\nNew Columns created from Email engineering are: {}'.format(newCols))
        print('\nEmail Engineering time taken: {}'.format(end-start))
    else:
        print('\nNew Columns created from URL engineering are: {}'.format(newCols))
        print('\nURL Engineering time taken: {}'.format(end-start)) 
    
    return return_df 



############################################
############## EMAIL URL ENGINEERING ##############
############################################