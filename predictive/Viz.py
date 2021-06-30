from collections import defaultdict
from sklearn.preprocessing import LabelEncoder
import pydotplus
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import re

# R packages
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri


#these imports can help you out if you use google colab to set up R env temporarily rather than have it in your environment

# utils = importr('utils')
# utils.install_packages('rpart',repos="https://mirror.niser.ac.in/cran/")
# utils.install_packages('rpart.plot',repos="https://mirror.niser.ac.in/cran/")
grdevices = importr('grDevices')
# utils.install_packages('rattle',repos="https://mirror.niser.ac.in/cran/")
# utils.install_packages('RColorBrewer',repos="https://mirror.niser.ac.in/cran/")


def cart_decisiontree(df, target_variable_name, class_or_Reg, priors,features_created):
    # removing the columns that were created during engineering
    df = df.copy()
    for col in features_created:
        if col in df.columns:
            df.drop(col,axis=1,inplace=True)
    # converting all category type columns to object type
    cat_df = df.select_dtypes('category')
    if not cat_df.empty:
        df.drop(cat_df.columns, axis=1, inplace=True)
        d = defaultdict(LabelEncoder)
        cat_df = cat_df.apply(lambda x: d[x.name].fit_transform(x.astype(str)))
        cat_df = cat_df.apply(lambda x: d[x.name].inverse_transform(x))
        df = pd.concat([df, cat_df], axis=1)

    for col in df.columns:
        if df[col].dtype == 'object':
            s = pd.Series(df[col])
            try:
                pd.to_numeric(s)
            except Exception as e:
                print(f"{col} column will now be truncated")
                df[col] = df[col].apply(
                    lambda x: x[0:6]+"." if (len(x) > 6) else x)
                print("Large text columns truncated")

    with localconverter(ro.default_converter + pandas2ri.converter):
        r_from_pd_df = ro.conversion.py2rpy(df)

    print("What you want", class_or_Reg)
    rstring1 = """
            function(data1){
            library(rpart)
            library(rpart.plot)
            library(rattle)
            library(RColorBrewer)
            fivepercent <- as.integer(0.05*nrow(data1))
            fit <- rpart("""+target_variable_name+"""~., data = data1,xval = 10,parms = priors,cp=0.001,maxdepth = 4,minsplit=fivepercent)
            rpart.plot(fit,roundint=TRUE)
            }
            """

    rstring2 = """
                function(data1){
                library(rpart)
                library(rpart.plot)
                library(rattle)
                library(RColorBrewer)
                fivepercent <- as.integer(0.05*nrow(data1))
                fit <- rpart("""+target_variable_name+"""~., data = data1,xval = 10,cp=0.001,maxdepth = 4,minsplit=fivepercent)
                rpart.plot(fit,roundint=TRUE)
                }
                """

    grdevices.svg(file="dec_tree.svg", width=16)

    # Front end needs this variable to resize the decision tree, pass it through the backend
    num_of_cols = len(df.columns)
    if class_or_Reg == 'Classification':
        rfunc = ro.r(rstring1)
        p = rfunc(r_from_pd_df)
    elif class_or_Reg == 'Regression':
        rfunc = ro.r(rstring2)
        p = rfunc(r_from_pd_df)
    grdevices.dev_off()
    from IPython.display import Image, display
    display(Image('dec_tree.svg'))
    print("CART Visualization Completed...")
