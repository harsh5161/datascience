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
import shap
import matplotlib.pyplot as plt
from imblearn.over_sampling import RandomOverSampler
import gc

def score(df, init_info, validation=False):
    '''
    Follows the same flow of data as there in INIT +  some plotting for ROS, Deciles + Interpretable Model Plots using Shap + generating the validation csv file
    '''
    print(">>>>>>[[VALIDATION AND SCORING]]>>>>>")

    if validation:
        X_train = init_info['X_train']
        y_train = init_info['y_train']

        if init_info['ML'] == 'Classification':
            priorList = y_train.value_counts(normalize=True).values
        else:
            priorList = None

    if validation:
        X_test = df.drop(init_info['Target'], axis=1)
        y_test = df[init_info['Target']]
    else:
        X_test = df
        y_test = pd.Series()

    if init_info['KEY']:
        df.dropna(axis=0, subset=[init_info['KEY']], inplace=True)
        kkey = df.dtypes[init_info['KEY']]
        try:
            if df[init_info['KEY']].dtype == np.float64:
                df[init_info['KEY']] = df[init_info['KEY']].astype(int)
        except:
            if kkey.any() == np.float64:             # if the key is float convert it to int
                df[init_info['KEY']] = df[init_info['KEY']
                                          ].iloc[:, 0].astype(int)
        if isinstance(df[init_info['KEY']], pd.DataFrame):
            k_test = df[init_info['KEY']].iloc[:, 0]
        else:
            k_test = df[init_info['KEY']]
        k_test.name = 'S.No'
        X_test = X_test.set_index(init_info['KEY'])
        k_test.index = X_test.index
    else:
        k_test = X_test.index
        k_test.name = 'S.No'

    lat = init_info['lat']
    lon = init_info['lon']
    lat_lon_cols = init_info['lat_lon_cols']
    if (lat and lon) or lat_lon_cols:
        LAT_LONG_DF = latlongEngineering(X_test, lat, lon, lat_lon_cols)
    else:
        LAT_LONG_DF = pd.DataFrame()

    date_cols = init_info['DateColumns']
    possible_datecols = init_info['PossibleDateColumns']
    if date_cols:
        DATE_DF = date_engineering(
            X_test[date_cols], possible_datecols, init_info['possibleDateTimeCols'], validation=True)
        DATE_DF = DATE_DF[init_info['DateFinalColumns']]
        DATE_DF.fillna(init_info['DateMean'], inplace=True)
    else:
        DATE_DF = pd.DataFrame()

    if init_info['EMAIL_STATUS'] is False:
        email_cols = init_info['email_cols']
        if len(email_cols) > 0:
            EMAIL_DF = emailUrlEngineering(
                X_test[email_cols], email=True, validation=True)
            EMAIL_DF.reset_index(drop=True)
        else:
            EMAIL_DF = pd.DataFrame()
    else:
        EMAIL_DF = pd.DataFrame()

    url_cols = init_info['url_cols']
    if len(url_cols) > 0:
        URL_DF = URlEngineering(X_test[url_cols])
        URL_DF.reset_index(drop=True)
    else:
        URL_DF = pd.DataFrame()

    X_test.reset_index(drop=True, inplace=True)
    DATE_DF.reset_index(drop=True, inplace=True)
    LAT_LONG_DF.reset_index(drop=True, inplace=True)
    EMAIL_DF.reset_index(drop=True, inplace=True)
    URL_DF.reset_index(drop=True, inplace=True)
    concat_list = [X_test, DATE_DF, LAT_LONG_DF, EMAIL_DF, URL_DF]
    X_test = pd.concat(concat_list, axis=1)

    if len(init_info['NumericColumns']) != 0:
        num_df = X_test[init_info['NumericColumns']]
        num_df = num_df.swifter.apply(
            lambda x: pd.to_numeric(x, errors='coerce'))
        num_df.fillna(init_info['NumericMean'], inplace=True)
    else:
        num_df = pd.DataFrame()

    if len(init_info['DiscreteColumns']) != 0:
        disc_df = X_test[init_info['DiscreteColumns']]
        disc_cat = init_info['disc_cat']
        for col in disc_df.columns:
            disc_df[col] = disc_df[col].apply(
                lambda x: x if x in disc_cat[col] else 'others')
        disc_df.fillna('missing', inplace=True)
    else:
        disc_df = pd.DataFrame()

    if init_info['remove_list'] is not None:
        X_test.drop(columns=init_info['remove_list'], axis=1, inplace=True)

    some_list = init_info['some_list']
    lda_models = init_info['lda_models']

    if some_list:
        start = time.time()
        sentiment_frame = sentiment_analysis(X_test[some_list])
        sentiment_frame.fillna(value=0.0, inplace=True)
        TEXT_DF = sentiment_frame.copy()
        TEXT_DF.reset_index(drop=True, inplace=True)
        end = time.time()
        start = time.time()
        new_frame = X_test[some_list].copy()
        new_frame.fillna(value="None", inplace=True)
        ind = 0
        for col in new_frame.columns:
            topic_frame, _ = topicExtraction(
                new_frame[[col]], True, lda_models['Model'][ind])
            topic_frame.rename(columns={0: str(col)+"_Topic"}, inplace=True)
            topic_frame.reset_index(drop=True, inplace=True)
            TEXT_DF = pd.concat([TEXT_DF, topic_frame], axis=1, sort=False)
            ind = ind+1
        X_test.drop(some_list, axis=1, inplace=True)
    else:
        TEXT_DF = pd.DataFrame()
    disc_df.reset_index(drop=True, inplace=True)
    num_df.reset_index(drop=True, inplace=True)
    TEXT_DF.reset_index(drop=True, inplace=True)
    if not TEXT_DF.empty:
        for col in TEXT_DF.columns:
            if col.find("_Topic") != -1:
                disc_df = pd.concat(
                    [disc_df, pd.DataFrame(TEXT_DF[col])], axis=1)
            else:
                num_df = pd.concat(
                    [num_df, pd.DataFrame(TEXT_DF[col])], axis=1)

    # Removing the same columns that were removed in Pearson's Correlation
    num_df = num_df[init_info['PearsonsColumns']]
    num_df.reset_index(drop=True, inplace=True)
    disc_df.reset_index(drop=True, inplace=True)
    if num_df.shape[1] != 0:  # Some datasets may contain only categorical data
        X_test = pd.concat([num_df, disc_df], axis=1)
    else:
        X_test = disc_df

    shapely_X = X_test.copy()
    shapely_X = shapely_X[init_info['TrainingColumns']]
    shapely_X = shapely_X.fillna(X_test.mode())
    print(">>>>>>[[Encoding Columns]]>>>>>")
    X_test = init_info['TargetEncoder'].transform(X_test)
    X_test = X_test[init_info['TrainingColumns']]
    X_test = X_test.fillna(X_test.mode())
    print(">>>>>>[[Scaling]]>>>>>")
    mm = init_info['MinMaxScaler']
    # Clip the data with training min and max, important
    X_test.clip(mm.data_min_, mm.data_max_, inplace=True, axis=1)
    X_test = mm.transform(X_test)
    X_test = pd.DataFrame(init_info['PowerTransformer'].transform(
        X_test), columns=init_info['TrainingColumns'])
    new_mm = MinMaxScaler()
    X_test = pd.DataFrame(new_mm.fit_transform(
        X_test), columns=init_info['TrainingColumns'])
    print(">>>>>>[[Applying Transformations]]>>>>>")
    if validation:
        if init_info['ML'] == 'Classification':
            ros = RandomOverSampler(sampling_strategy='minority',random_state=42)
            X_rt, y_rt = ros.fit_resample(X_test, y_test)
        else:
            X_rt, y_rt = X_test, y_test

    if validation:
        start = time.time()
        gc.collect()
        ############# MODEL TRAINING #############
        print("\n\n")
        print(">>>>>>[[MODELLING]]>>>>>")
        mod, model_info, exp_mod, exp_name, feat_mod, feat_name = model_training(
            X_train, y_train, X_test, y_test, init_info['ML'], priorList, init_info['q_s'])
        rule_result = ruleTesting(
            X_rt, y_rt, init_info['ML'], init_info['rule_model'], init_info['TargetLabelEncoder'],init_info['features_created'])
        print('MODEL SAVED')

        ############# MODEL TRAINING #############
        end = time.time()
        print('\nTotal Model Training Time taken : {}'.format(end-start))
        print(">>>>>>[[Modelling Completed]]>>>>>")
        gc.collect()
        ############# PREDICTION/SCORING #############
    else:
        mod = init_info['model']
    y_pred = mod.predict(X_test)

    if validation:
        regplotdf = pd.DataFrame()
        regplotdf['y_test'] = y_test
        regplotdf['y_pred'] = y_pred

    if init_info['ML'] == 'Classification':
        y_probas = mod.predict_proba(X_test)
        y_pred = pd.Series(
            init_info['TargetLabelEncoder'].inverse_transform(y_pred))
        if validation:
            try:
                y_test = pd.Series(
                    init_info['TargetLabelEncoder'].inverse_transform(y_test))
            except IndexError:
                print("Index error triggered")
                y_test = pd.Series(
                    init_info['TargetLabelEncoder'].inverse_transform(y_test.astype(int)))
            y_probs_cols = ['Class ' +
                            str(x) + ' Probabilities' for x in init_info['TargetLabelEncoder'].inverse_transform(mod.classes_)]
            init_info['y_probs_cols'] = y_probs_cols
        else:
            y_probs_cols = init_info['y_probs_cols']
        
            
        y_probas = pd.DataFrame(y_probas, columns=y_probs_cols)

        if validation:
            from sklearn.metrics import classification_report
            print(classification_report(y_test, y_pred))

            axcm = plt.subplot()
            skplt.metrics.plot_confusion_matrix(
                y_test, y_pred, cmap='RdPu', ax=axcm)
            axcm.set_xlabel('Predicted value')
            axcm.set_ylabel('Actual value')
            if len(priorList) == 2:

                skplt.metrics.plot_lift_curve(
                    y_test, y_probas, title='Lift Curve (Test Dataset)')
                skplt.metrics.plot_cumulative_gain(
                    y_test, y_probas, title='Cumulative Gains Curve (Test Dataset)')
            plt.close('all')
    else:
        if validation:
            import seaborn as sns
            import math
            from tabulate import tabulate

            # residual plot
            fig1 = sns.residplot('y_test', 'y_pred', regplotdf)
            plt.xlabel("Actual Values")
            plt.ylabel("Residuals of Predicted Values")
            plt.title("\n\nResidual Plot (Test Dataset)")
            plt.show(fig1)
            plt.close()
            # lm plot
            fig2 = sns.lmplot('y_pred', 'y_test', regplotdf,
                              fit_reg=True, line_kws={'color': 'red'})
            plt.xticks(plt.xticks()[0], rotation=40)
            plt.xlabel("Predicted Values")
            plt.ylabel("Actual Values")
            plt.title("\n\nPredicted vs Actual (Test Dataset)")
            plt.show(fig2)
            plt.close()
            # decile plot function

            def decileplot(regplotdf):
                div = math.floor(len(regplotdf)/10)
                sorted_df = pd.DataFrame(
                    regplotdf.sort_values('y_test', ascending=False))
                sorted_df['decile'] = 0
                for i in range(1, 11):
                    sorted_df.iloc[div*(i-1):div*i, 2] = i
                sorted_df = sorted_df[sorted_df.decile != 0]
                df_mean = pd.DataFrame()
                df_mean[['Decile', 'Actualvalue_mean', 'Predictedvalue_mean']] = sorted_df.groupby(
                    'decile', as_index=False)[['y_test', 'y_pred']].mean()
                df_mean['Actualvalue_mean'] = pd.Series(
                    df_mean['Actualvalue_mean']).round(decimals=2)  # rounding off values
                df_mean['Predictedvalue_mean'] = pd.Series(
                    df_mean['Predictedvalue_mean']).round(decimals=2)  # rounding off values
                fig, ax1 = plt.subplots(figsize=(10, 7))
                plt.xticks(df_mean['Decile'])
                tidy = pd.melt(df_mean, id_vars='Decile', value_vars=[
                               'Actualvalue_mean', 'Predictedvalue_mean'], value_name='Mean values per decile')
                sns.lineplot(x='Decile', y='Mean values per decile', hue='variable', data=tidy, ax=ax1).set_title(
                    'Distribution of Actual vs Predicted Values in the Test Dataset by Deciles')

                def pdtabulate(df): return tabulate(
                    df, headers='keys', tablefmt='psql', showindex=False)
                print(
                    "\nDistribution of Mean of Actual and Predicted Values by Deciles:")
                print(pdtabulate(df_mean))

            # decile plot
            decileplot(regplotdf)
            plt.close('all')
    ############ PREDICTION/SCORING #############

    ############ Model Explainer#############
    if validation:
        r = random.random()
        b = random.random()
        g = random.random()
        colors = (r, g, b)

        # customized number
        if len(X_test.columns) > 10:
            num_features = 10
        else:
            num_features = len(X_test.columns)

        features = X_test.columns
        try:
            top_ten = featureimportance(
                feat_mod, feat_name, num_features, features)
        except:
            print(feat_name,"did not work")
            try:
                top_ten = featureimportance(
                    exp_mod, exp_name, num_features, features)
            except:
                print(exp_name,"did not work")
                try:
                    top_ten = featureimportance(
                            init_info['feature_selection_model'], 'LightGBM Tree Classifier', num_features, init_info['feature_selection_features'])
                except Exception as e:  
                    print(f"{e}")

        shap.initjs()
        if len(X_test) > 20000:
            samp = X_test.sample(n=20000, axis=0,random_state=42)
            samp_X = shapely_X.sample(n=20000, axis=0,random_state=42)
        else:
            samp = X_test
            samp_X = shapely_X
        try:
            explainer = shap.TreeExplainer(exp_mod, data=samp)
            shap_values = explainer.shap_values(
                samp, tree_limit=10, check_additivity=False)
            if init_info['ML'] == 'Regression':
                for i in range(0, 4):
                    value = int(np.random.randint(0, len(shap_values), 1))
                    shap.force_plot(explainer.expected_value, shap_values[value, :], samp_X.iloc[value, :], matplotlib=True, show=False).savefig(
                        f'forceplot{i}.png', bbox_inches='tight')
        except:
            try:
                explainer = shap.Explainer(exp_mod.predict, samp)
                shap_values = explainer.shap_values(
                    samp, check_additivity=False)
                if init_info['ML'] == 'Regression':
                    for i in range(0, 4):
                        value = int(np.random.randint(0, len(shap_values), 1))
                        shap.force_plot(explainer.expected_value, shap_values[value, :], samp_X.iloc[value, :], matplotlib=True, show=False).savefig(
                            f'forceplot{i}.png', bbox_inches='tight')
            except Exception as e:
                print(f"{e} : {exp_name} Model type not supported by SHAP.")
        plt.close('all')
        try:
            print(f'shap value shape is as follows {len(shap_values)}')
            try:
                for i in range(len(shap_values)):
                    # Use exp_name variable to get
                    string = f'Summary plot of {exp_name}'
                    shap.summary_plot(shap_values[i], samp, title=string, class_names=list(
                        init_info['TargetLabelEncoder'].classes_))
                    break
            except Exception as e:
                print(e)
                # Use exp_name variable to get
                string = f'Summary plot of {exp_name}'
                shap.summary_plot(shap_values, samp, title=string)

            if init_info['ML'] == 'Classification':
                LE = init_info['TargetLabelEncoder']
                if len(LE.classes_) > 2:
                    maps = np.flipud(LE.transform(LE.classes_))
                    le_mapping = dict(zip(LE.classes_, maps))
                else:
                    le_mapping = dict(
                        zip(LE.classes_, LE.transform(LE.classes_)))
                encoded_targ = pd.DataFrame(le_mapping.items(), columns=[
                                            'Target Classes', 'Encodings'])
            for top in top_ten:
                for idf in init_info['encoded_disc']:
                    if top in idf.columns.tolist()[0]:
                        # print(idf) # for webapp use
                        pass
                    else:
                        continue
        except Exception as e:
            print(e)

    plt.close('all')
    ############ Model Explainer #############
    ############ PREDICTION/SCORING #############

    if validation:
        mc = model_info.drop(['model', 'param', 'accuracy'], axis=1)
        if init_info['ML'] == 'Classification':
            mc.sort_values('Weighted F1', ascending=False, inplace=True)
        else:
            mc.sort_values('RMSE', ascending=True, inplace=True)
        saved_mc = mc.copy()
        for col in saved_mc.columns:
            if col not in ['Machine Learning Model', 'Accuracy%', 'Total time (hh:mm:ss)']:
                saved_mc[col] = round(mc[col].astype(float), 3)
        saved_mc.to_csv('MC.csv', index=False)
        # This removes the data from dict to avoid storage
        del init_info['X_train'], init_info['y_train']
        init_info['model'] = mod
        joblib.dump(init_info, 'model_info', compress=9)

    preview_length = 100 if len(X_test) > 100 else len(X_test)
    if validation:
        preview = pd.DataFrame({k_test.name: k_test.values.tolist(),
                                'Actual Values': y_test.tolist(),
                                'Predicted Values': y_pred.tolist()})
    else:
        preview = pd.DataFrame({k_test.name: k_test.values.tolist(),
                                'Predicted Values': y_pred.tolist()})

    if init_info['ML'] == 'Classification':
        preview = pd.concat([preview, y_probas], axis=1)
        if validation:
            # to convert '1.0' and '0.0' to '1' and '0'
            for col in ['Actual Values', 'Predicted Values']:
                if preview[col].dtype == np.float64:
                    preview[col] = preview[col].astype(int)

        else:
            # to convert '1.0' and '0.0' to '1' and '0'
            for col in ['Predicted Values']:
                if preview[col].dtype == np.float64:
                    preview[col] = preview[col].astype(int)

        yp = {}
        for i in y_probas.columns:
            yp[i] = str(i).replace("Probabilities", "Probability")
            yp[i] = str(i).replace("0.0", "0")
            yp[i] = str(i).replace("1.0", "1")
        preview.rename(columns=yp, inplace=True)       # to rename columns

    for col in preview.columns:       # to round off decimal places of large float entries in preview
        if preview[col].dtype == np.float64:
            preview[col] = pd.Series(preview[col]).round(decimals=3)

    if validation:
        sort_col = preview['Predicted Values']
        try:
            preview, _ = train_test_split(
                preview, train_size=preview_length, random_state=42, stratify=sort_col)
        except:
            preview = preview[:preview_length]

        preview_vals = preview['Predicted Values'].value_counts()
        printer = ""

        for k, v in preview_vals.iteritems():
            printer = printer + \
                f"{k} is present in {v}% of the Testing Preview\n"

        preview.to_csv('preview.csv', sep=',', index=False)
        print(">>>>>>[[Preview Saved]]>>>>>")
    else:
        preview_vals = preview['Predicted Values'].value_counts()
        printer = ""
        for k, v in preview_vals.iteritems():
            printer = printer + \
                f"{k} is present in {round((v/len(preview))*100,3)}% of the Scoring File\n"
        preview.to_csv('score.csv', sep=',', index=False)
        print(">>>>>>[[Scoring File Saved]]>>>>>")
    print('\nCode executed Successfully')
    print(">>>>>>[[Validation and Scoring Completed]]>>>>>")
