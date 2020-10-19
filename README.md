# Logical Proceeding of ProtonAutoML (v. 1.0)

Flow of informations starts from Training <br />
As the data enters the system, we check the column names for special characters and we remove them. This is done because the Light GBM ML model does not support<br /> special characters in the column names.<br />
<br />
A random sample of 1000 rows of the original dataset is then taken and we perform numeric engineering on this sample, we do this to call the targetAnalysis() <br />that is used to check if the target variable is “Classification” or “Regression” or “None”. Then we cap the dataset at a maximum length of 10,00,000 rows.<br /> Stratification done for Classification Problems and Random Sampling done for Regression problems.<br />
<br />
Now the Flow of information moves into INIT<br />
We print the list of MISSING percentages present in every column to study the data.<br />
<br />
# Numeric Engineering<br />
Pass everything through numeric engineering - Strip spaces, symbols, lower case all entries, replacing empty and invalid strings, replacing comma when <br />currencies, removing % after a value, removing dollar before a value.<br />
Note that Numeric Engineering of Target  variable happens separately after the above.<br />
<br />
# targetAnalysis() - <br />
If it is numeric and more than 5 levels, send it for regression modelling, if it's 5 levels or less, send it for classification modelling. <br />
If it's categorical and if the number of levels is less than 5, just send for classification modelling. <br />
<br />
# removeLowClass() -<br />
If it's a binary classification problem then we don't remove any of the classes.<br />
If there is only one class present, the target variable is not in a state to apply classification or regression because prediction cannot be done.<br />
In a multiclass problem, we replace the classes that account for less than 0.5% of the length of the dataset and we replace those classes with np.nan which is <br />later dropped.<br />
<br />
# UserInteractVisualisation()-<br />
Numeric columns(dtype int64 and float64) are used to plot Histograms.<br />
Object columns are considered for bivariate plotting:<br />
If the top 10 unique values in an object column does not account for more than 10% of the data then that column is not considered for bivariate plotting. <br />
Otherwise we call the bivar_plotter() function.<br />
<br />
# Bivariateploter()-<br />
# This function's logic is shoddy. Need to be looked at and also revamped.<br />
<br />
<br />
The INIT then removes any row that contains a Null value in the target column.<br />
The dataset is then into training test and testing set(named as “validation” set in the code, which is a misnomer as a holdout validation set is not actually <br />being used to evaluate the performance of the training models)<br />
<br />
# DatasetSelection()-<br />
We are trying to minimise the loss of information here by utilising two different methods primarily.<br />
We either FIRST drop rows that contain more than 50% missing columns and then subsequently drop columns that contain more than 50% missing rows.<br />
OR<br />
We drop columns that contain more than 50% missing rows and then subsequently drop rows that contain more than 50% missing columns.<br />
<br />
We check which of these methods may result in a less loss of rows and that is the method chosen for that particular dataset.<br />
<br />
# Date Engineering<br />
Support many formats, reads data (including date time functions).<br />
All non numeric columns are sent into this function.<br />
If the name of the column contains “Date” in it then the Possibility of that column being a date is set as the 10% of the length of the dataset, otherwise it is set as 0.<br />
Now the possibilities of all of the object columns are increased by one for every entry in that column IF:<br />
The date is split with ‘/’,’-’ or’:’<br />
If there is a month in words present in that entry.<br />
Otherwise the possibility is decreased by 1.<br />
<br />
If the possibility number of a column accounts for greater than 80% of the dataset length then it is definitely a column.<br />
In the other columns we check if month in words is present in at least 80% of the values, if it is then we consider that column a “Possibly a Date Column”.<br />
<br />
In the list of “Possibly a Date Column” we check if a particular column is of the form “12-10-2020:00:00:00”, if so we force convert that column into a date<br /> format that works with our system. (check messy2 dataset)<br />
<br />
If a recognised date column contains more than 35% null values then we drop that column.<br />
We are imputing the date columns using the mode of that particular column when possible and when not possible we are imputing it with the mean so that the <br />imputation remains true to the sample.<br />
<br />
Columns created are - difference of all date columns, date_month, date_year, date-today, date-nearestholiday (often spike in sales is during holidays). <br />
<br />
<br />

# Segregation() and Outlier Winsorizing-<br />
Numeric columns are separated into numeric or categorical numeric. Numeric variables with less than 8 levels are considered categorical numeric.<br />
<br />
Grouping of categorical data - if the top 5 levels (by number of observations) in a variable have more than a total of 10% of observations, we send it for<br /> grouping, else it's called a useless column. After checking this, we group levels having less than 0.5% value as ‘others’. After all this if a variable is more<br /> than 60 levels, we reject it.<br /> 
<br />
Outlier winsorizing = bring back the extreme values to within +-4sd <br />
<br />
#Text Engineering-<br />
Only Useless columns are considered in Text Engineering.<br />
If there are fewer than 5 words in more than 75% of the rows then that particular row is dropped because it is probably not a review column.<br />
<br />
Each column that is not dropped is tokenised. Every value present in that column token is called an entity. We use spaCy’s pretrained models to check if a<br /> particular entity present in the column token is a countries, cities, state, person (includes fictional), nationalities, religious groups, buildings, airports,<br /> highways, bridges, companies, agencies, institutes, DATE etc.<br />
We then count the number of entities present in a column token that match  the above criteria. If the match is greater than 60% of the total number of entities <br />present in that column token then we drop that column because it is probably not a review column.<br />
<br />
On the columns that are recognised that a review column, we perform the following activities.<br />
Sentiment Analysis is performed to generate “Review-Polarity” and “Review-Subjectivity” columns. Null values present in these columns are imputed to 0.0 to <br />maintain neutrality.<br />
Topic Extraction is performed to generate 10 topics for every Review Column found and allocate a topic number to those 10 topics and add it as “Review-Topic”<br /> column.<br />
Topics are extracted by first lemmatising and preprocessing the review columns by removing stop words, converting into lower case strings, and then tokenised and<br /> converted into a dictionary. We then select the most frequent tokens that are present in at least 25% of the documents at least, with a lower limit of<br /> the token being present in at least 10 documents and we select the most frequent 1000 tokens. These are then converted into a Bag of Words corpus.<br />
<br />
The BOW corpus is then used to generate a Tf-Idf corpus for the review column. We generate the topics using gensim’s LDA model and the Tf-Idf generated. The <br />Review column is dropped and replaced with the three columns generated by text engineering.<br />
<br />
<br />
# Pearson’s Correlations and other Target Encoding<br />
 we check pearson correlation of all variables, remove the highest, then check again and remove the highest until the threshold is met (0.85 i think) <br />
# Why is only the upper limit of 0.85 dropped and not the negative limit of -0.85? <br />
<br />
Target encoding - target encode all categorical variables <br />
<br />
# FeatureSelection and Plots-<br />
Light GBM is used as the base estimator for feature selection. Class weight balancing is done to handle moderate imbalance in the data.<br />
Two thresholds are used in selecting important features, the first one is very restrictive so it returns very few features. From the remaining features a mean is<br /> computed and used as a threshold to find more features that are important.<br />
Prints top 15 variables.<br />
<br />
<br />
# CART visualization<br />
The decision tree visualisation is written in R and wrapped to Python using the RPy2 module. <br />
The Sklearn version of the decision tree is still present and not yet deprecated from code.<br />
<br />
# Normalization and Transformations<br />
MinMax scaling and Box-Cox Power Transformations are done to standardise and transform the data into Gaussian Distribution.<br />
<br />
# Sample Equation<br />
For classification problems, the sample equation is generated using Logistic Regression.<br />
For regression problems, the sample equation is generated using Linear Regression with Sequential Feature Selector.<br />
The object columns that are encoded in the sample equation are displayed for the users better understanding.<br />
<br />
# Now the Flow of Information moves into SCORE.py<br />
The testing set and the scoring set both undergo the same preprocessing steps as discussed above for the training set. Then we move into model building.<br />
<br />
# Modelling.py<br />
If the user wants quick results then we run default machine learning models with pre defined parameters.<br />
If a user wants slow results, then we perform hyperparameter optimisation using hyperopt.<br />
The hyperopt function performs a Bayesian Optimisation technique to fit the hyperparameters onto the training set with cross validation.<br />
<br />
The models used in Classification problems are as follows:<br />
XGBoost<br />
CATBoost<br />
LightGBM<br />
Random Forest<br />
ExtraTrees <br />
Naive Bayes<br />
Logistic Regression<br />
Neural Net (MLP sklearn)<br />
SVC<br />

In case of heavy imbalance, i.e if the minority class is less than 5% of the total length of the dataset in the case of binary classification problems, the <br />aforementioned algorithms are not used and instead the following are used at predefined parameters:<br />
ADABoost Ensemble Estimator<br />
LightGBM Ensemble Estimator<br />
XGBoost Ensemble Estimator<br />
Random Forest Ensemble Estimator<br />
In Binary Classification, in the class distribution if one of the classes accounts for less than 25% of the dataset then Neural Net and Naive Bayes will not get<br /> executed from the main classification algorithms list, this is to avoid executing poor performing algorithms that take a lot of time to execute.<br />
For Multiclass Classification, the threshold is 15% for the nullification for Neural Nets and Naive Bayes<br />
<br />
After the models are built, poor performing models are dropped from the ensemble. The poor performing models are judged on the basis of the F1 score of a <br />particular classifier on the classes. <br />
<br />
The remaining models are used to create an ensemble learner using a VotingClassifier.<br />
In classification the evaluation metric used to judge the performance is the Weighted F1 Score.<br />
<br />
In Regression problems, the algorithms that are used are as follows:<br />
XGBoost<br />
CATBoost<br />
Light GBM<br />
Random Forest<br />
Extra Trees<br />
Linear Regression<br />
Ridge Regression<br />
Neural Network<br />
SVR<br />
<br />
Similar to Classification, an ensemble learner is created using VotingRegressor.<br />
The performance of all generated regression models are assessed using the RMSE value.<br />
The scores present in the Model Info table is generated by testing on the test set.<br />
<br />
The final best model, both in Classification as well as Regression, is saved to be used for Scoring.<br />
<br />
<br />
# Additional Plotting-<br />
For all classification Problems Confusion Matrix and ROC_AUC curve is plotted.<br />
Additionally, specifically for Binary Classification problems, Lift curve and Cumulative gains chart is also plotted.<br />
<br />
For all regression Problems Residual Plot, LMPlot and Decile Plots are generated.<br />
<br />
# Scoring<br />
After modelling is generated, user can input a scoring dataset to generate predictions. The entire file is automatically downloaded.<br />
Say,  the training dataset has features ranging from X1 to X10, and if the Scoring dataset contains features ranging from X1 to X20. The Scoring will still be<br /> done as long as the variables that are required for training are present in the scoring set.<br />
<br />
# Preview<br />
A Preview of the testing predictions is generated. This Preview contains 100 rows and is stratified for classification problems (i.e will contain enough <br />representation of minority predictions), however for regression problems a random sample of 100 rows from the testing predictions is generated.<br />

<br />






