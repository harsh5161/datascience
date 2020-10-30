from modelling import *
from engineerings import *
from all_other_functions import *
import pandas as pd
import numpy as np
import scikitplot as skplt
import seaborn as sns
import swifter
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.model_selection import train_test_split

def score(df,init_info,validation=False):
    print('\n\t #### VALIDATION AND SCORING ZONE ####')

    if validation:
        X_train = init_info['X_train']
        y_train = init_info['y_train']

        if init_info['ML'] == 'Classification':
            priorList = y_train.value_counts(normalize=True).values
        else:
            priorList = None

    if validation:
        X_test = df.drop(init_info['Target'],axis=1)
        y_test = df[init_info['Target']]
    else:
        X_test = df
        y_test = pd.Series()

    if init_info['KEY']:
        print("Value of Key from Training is: ",init_info['KEY'])
        print("Total no. of null values present in the Key: ",df[init_info['KEY']].isna().sum())
        df.dropna(axis=0,subset=[init_info['KEY']],inplace=True)
        print("NUll values after removal are: ",df[init_info['KEY']].isna().sum())
        if df[init_info['KEY']].dtype == np.float64:             # if the key is float convert it to int
            df[init_info['KEY']]=df[init_info['KEY']].astype(int)        
        k_test = df[init_info['KEY']]
        k_test.index = X_test.index
    else:
        k_test = X_test.index
        k_test.name = 'S.No'

    date_cols = init_info['DateColumns']
    possible_datecols= init_info['PossibleDateColumns']
    if date_cols:
        DATE_DF = date_engineering(X_test[date_cols], possible_datecols, validation=True)
        DATE_DF = DATE_DF[init_info['DateFinalColumns']]
        DATE_DF.fillna(init_info['DateMean'],inplace=True)
    else:
        DATE_DF = pd.DataFrame()

    if len(init_info['NumericColumns'])!=0:
        num_df = X_test[init_info['NumericColumns']]
        num_df = num_df.swifter.apply(lambda x : pd.to_numeric(x,errors='coerce'))
        num_df.fillna(init_info['NumericMean'],inplace=True)
    else:
        num_df = pd.DataFrame()

    if len(init_info['DiscreteColumns'])!=0:
        disc_df = X_test[init_info['DiscreteColumns']]
        disc_cat = init_info['disc_cat']
        for col in disc_df.columns:
            disc_df[col] = disc_df[col].apply(lambda x: x if x in disc_cat[col] else 'others')
        disc_df.fillna('missing',inplace=True)
    else:
        disc_df = pd.DataFrame()

    if init_info['remove_list'] is not None:
        X_test.drop(columns=init_info['remove_list'],axis=1,inplace=True)

    some_list = init_info['some_list']
    lda_models = init_info['lda_models']

    if some_list:
        print("The review/comment columns found are", some_list)
        start = time.time()
        sentiment_frame = sentiment_analysis(X_test[some_list])
        sentiment_frame.fillna(value=0.0,inplace=True)
        print(sentiment_frame)
        TEXT_DF = sentiment_frame.copy()
        TEXT_DF.reset_index(drop=True,inplace=True)
        end = time.time()
        print("Sentiment time",end-start)
        start = time.time()
        new_frame = X_test[some_list].copy()
        new_frame.fillna(value="None",inplace=True)
        ind = 0
        for col in new_frame.columns:
            topic_frame, _ = topicExtraction(new_frame[[col]],True,lda_models['Model'][ind])
            topic_frame.rename(columns={0:str(col)+"_Topic"},inplace=True)
            print(topic_frame)
            topic_frame.reset_index(drop=True, inplace=True)
            TEXT_DF = pd.concat([TEXT_DF, topic_frame], axis=1, sort=False)
            ind = ind+1
        X_test.drop(some_list,axis=1,inplace=True)
    else:
        TEXT_DF = pd.DataFrame()


    print('\n #### TRANSFORMATION AND PREDICTION ####')
    num_df.reset_index(drop=True, inplace=True)
    disc_df.reset_index(drop=True, inplace=True)
    DATE_DF.reset_index(drop=True, inplace=True)
    TEXT_DF.reset_index(drop=True, inplace=True)
    print('num_df - {}'.format(num_df.shape))
    print('disc_df - {}'.format(disc_df.shape))
    print('DATE_DF - {}'.format(DATE_DF.shape))
    print('TEXT_DF - {}'.format(TEXT_DF.shape))
    X_test = pd.concat([num_df,disc_df,DATE_DF,TEXT_DF],axis=1)
    X_test = init_info['TargetEncoder'].transform(X_test)
    X_test = X_test[init_info['TrainingColumns']]
    X_test = X_test.fillna(X_test.mode())
    mm = init_info['MinMaxScaler']
    X_test.clip(mm.data_min_,mm.data_max_,inplace=True,axis=1) #Clip the data with training min and max, important
    X_test = mm.transform(X_test)
    X_test = pd.DataFrame(init_info['PowerTransformer'].transform(X_test),columns=init_info['TrainingColumns'])
    new_mm = MinMaxScaler()
    X_test = pd.DataFrame(new_mm.fit_transform(X_test),columns=init_info['TrainingColumns'])
    print('\nThis is final shape of X_test : {}'.format(X_test.shape))

    # joblib.dump(X_test,'Xt')
    # joblib.dump(X_train,'XT')
    # joblib.dump(y_train,'YT')
    # joblib.dump(y_test,'Yt')

    print('\n #### PRINTING THE LIST OF COLUMNS AND ITS TYPES THAT ENTER THE MODEL TRAINING ####')
    print('#### PRINTING X_test ####')
    print(X_test.columns)
    print(X_test.dtypes)
    print('\n')
    print(X_test.head(20))
    print('\n\n')
    if validation:
        print('#### PRINTING X_train ####')
        print(X_train.columns)
        print(X_train.dtypes)
        print('\n')
        print(X_train.head(20))
        print('\n\n')
        start = time.time()
        ############# MODEL TRAINING #############
        mod,model_info = model_training(X_train,y_train,X_test,y_test,init_info['ML'],priorList,init_info['q_s'])
        print('MODEL SAVED')
        ############# MODEL TRAINING #############
        end = time.time()
        print('\nTotal Model Training Time taken : {}'.format(end-start))
        ############# PREDICTION/SCORING #############
    else:
        mod = init_info['model']
    y_pred = mod.predict(X_test)

    if validation:
        regplotdf=pd.DataFrame()
        regplotdf['y_test']=y_test
        regplotdf['y_pred']=y_pred

    if init_info['ML'] == 'Classification':
#         y_probas = xg.predict_proba(X_test)
        y_probas = mod.predict_proba(X_test)
        y_pred = pd.Series(init_info['TargetLabelEncoder'].inverse_transform(y_pred))
        if validation:
            y_test = pd.Series(init_info['TargetLabelEncoder'].inverse_transform(y_test))
            y_probs_cols = ['Class ' + str(x) +' Probabilities' for x in y_pred.unique()]
            init_info['y_probs_cols'] = y_probs_cols
        else:
            y_probs_cols = init_info['y_probs_cols']
        # print("!!!!!!!!!!!!!!!!!!!!!!!")
        # print("YPROBAS IS AS FOLLOWS",y_probas)
        # print("YPRED IS AS FOLLOWS",y_pred)
        # print("YPROBSCOLUMNS ARE AS FOLLOWS",y_probs_cols)
        # print("!!!!!!!!!!!!!!!!!!!!!!!")
        y_probas = pd.DataFrame(y_probas,columns=y_probs_cols)

        if validation:
            from sklearn.metrics import classification_report
            print(classification_report(y_test,y_pred))

            axcm= plt.subplot()
            skplt.metrics.plot_confusion_matrix(y_test, y_pred, cmap= 'RdPu', ax =axcm)
            axcm.set_xlabel('Predicted value');
            axcm.set_ylabel('Actual value');
            if len(priorList) ==2:

                skplt.metrics.plot_lift_curve(y_test, y_probas)
                skplt.metrics.plot_cumulative_gain(y_test, y_probas)
            skplt.metrics.plot_roc(y_test, y_probas)

    else:
        if validation:
            import seaborn as sns
            import math
            from tabulate import tabulate

            #residual plot
            # fig1 = sns.residplot('y_test','y_pred',regplotdf)
            # plt.xlabel("Actual Values")
            # plt.ylabel("Residuals of Predicted Values")
            # plt.title("\n\nResidual Plot")
            # plt.show(fig1)

            #lm plot
            # fig2 = sns.lmplot('y_pred','y_test',regplotdf,fit_reg =True, line_kws={'color': 'red'})
            # plt.xlabel("Predicted Values")
            # plt.ylabel("Actual Values")
            # plt.title("\n\nPredicted vs Actual")
            # plt.show(fig2)

            # decile plot function
            def decileplot(regplotdf):
                div=math.floor(len(regplotdf)/10)
                sorted_df= pd.DataFrame(regplotdf.sort_values('y_test',ascending=False))
                sorted_df['decile']=0
                for i in range(1,11):
                    sorted_df.iloc[div*(i-1):div*i,2]= i
                sorted_df = sorted_df[sorted_df.decile != 0]
                df_mean=pd.DataFrame()
                df_mean[['Decile','Actualvalue_mean','Predictedvalue_mean']]=sorted_df.groupby('decile', as_index=False)[['y_test','y_pred']].mean()
                df_mean['Actualvalue_mean']= pd.Series(df_mean['Actualvalue_mean']).round(decimals=2) #rounding off values
                df_mean['Predictedvalue_mean']= pd.Series(df_mean['Predictedvalue_mean']).round(decimals=2) #rounding off values
                fig, ax1 = plt.subplots(figsize=(10, 7))
                plt.xticks(df_mean['Decile'])
                tidy = pd.melt(df_mean, id_vars='Decile', value_vars= ['Actualvalue_mean','Predictedvalue_mean'],value_name='Mean values per decile')
                sns.lineplot(x='Decile', y='Mean values per decile', hue='variable', data=tidy, ax=ax1)
                pdtabulate=lambda df:tabulate(df,headers='keys',tablefmt='psql', showindex = False)
                print("\nDistribution of Mean of Actual and Predicted Values by Deciles:")
                print(pdtabulate(df_mean))

            # decile plot
            decileplot(regplotdf)

            # y_probas = pd.Series()
            # fig3 = plt.figure()
            # plt.plot(y_pred, figure =fig3)
            # plt.plot(np.ones(len(y_pred))*y_pred.mean(), figure=fig3)
            # plt.show()
    ############ PREDICTION/SCORING #############
    if validation:
        mc = model_info.drop(['model','param','accuracy'],axis=1)
        if init_info['ML'] == 'Classification':mc.sort_values('Weighted F1',ascending=False,inplace=True)
        else:mc.sort_values('RMSE',ascending=True,inplace=True)
        mc.to_csv('MC.csv',index=False)
        del init_info['X_train'],init_info['y_train']                  # This removes the data from dict to avoid storage
        init_info['model'] = mod
        joblib.dump(init_info,'model_info',compress=9)

    preview_length = 100 if len(X_test)>100 else len(X_test)
    if validation:
        preview = pd.DataFrame({k_test.name:k_test.tolist(),
                                'Actual Values':y_test.tolist(),
                                'Predicted Values':y_pred.tolist()})
    else:
        preview = pd.DataFrame({k_test.name:k_test.tolist(),
                                'Predicted Values':y_pred.tolist()})

    if init_info['ML'] == 'Classification':
        preview = pd.concat([preview,y_probas],axis=1)
        if validation:
            for col in ['Actual Values','Predicted Values']:                # to convert '1.0' and '0.0' to '1' and '0'
                if preview[col].dtype== np.float64:
                    preview[col]=preview[col].astype(int) 
        else:
            for col in ['Predicted Values']:                # to convert '1.0' and '0.0' to '1' and '0'
                if preview[col].dtype== np.float64:
                    preview[col]=preview[col].astype(int) 
                        
        yp={}
        for i in y_probas.columns:
            yp[i] = str(i).replace("Probabilities", "Probability")
            yp[i] = str(i).replace("0.0", "0")
            yp[i] = str(i).replace("1.0", "1")
        preview.rename(columns = yp, inplace = True)       # to rename columns


    for col in preview.columns:       # to round off decimal places of large float entries in preview
            if preview[col].dtype == np.float64:
                preview[col]= pd.Series(preview[col]).round(decimals=3)
            
    if validation:
        sort_col = preview['Predicted Values']
        try:
        	preview, _ = train_test_split(preview,train_size=preview_length,random_state=1,stratify=sort_col)
        except:
        	preview = preview[:preview_length]
        
        preview_vals = preview['Predicted Values'].value_counts()
        printer = ""
        for k,v in preview_vals.iteritems():
            printer = printer + f"{k} is present in {v}% of the Testing Preview\n"
        print(printer)
        preview.to_csv('preview.csv',sep=',',index=False)
        print('\nFile Saved as preview.csv')
    else:
        preview_vals = preview['Predicted Values'].value_counts()
        printer = ""
        for k,v in preview_vals.iteritems():
            printer = printer + f"{k} is present in {round((v/len(preview))*100,3)}% of the Scoring File\n"
        print(printer)
        preview.to_csv('score.csv',sep=',',index=False)
        print('\nFile Saved as score.csv')
    print('\nCode executed Successfully')
    print('\n############# END ###########')
