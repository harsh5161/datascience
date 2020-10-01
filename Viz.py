import pydotplus
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import re

#R packages
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri

# utils = importr('utils')
# utils.install_packages('rpart.plot',repos="https://mirror.niser.ac.in/cran/")
grdevices = importr('grDevices')
# utils.install_packages('rattle',repos="https://mirror.niser.ac.in/cran/")
# utils.install_packages('RColorBrewer',repos="https://mirror.niser.ac.in/cran/")

def cart_decisiontree(df,target_variable_name,class_or_Reg,priors):
	with localconverter(ro.default_converter + pandas2ri.converter): 
		r_from_pd_df = ro.conversion.py2rpy(df)
		# class_or_Reg = ro.conversion.py2rpy(typer)
		
	print("What you want",class_or_Reg)		
	rstring1="""
			function(data1){
			library(rpart)
			library(rattle)
			library(rpart.plot)
			library(RColorBrewer)
			fit <- rpart("""+target_variable_name+"""~., data = data1,xval = 10,parms = priors,cp=0,maxdepth = 4,minsplit=6)
			rpart.plot(fit,roundint=TRUE)
			}
			"""

	rstring2="""
				function(data1){
				library(rpart)
				library(rattle)
				library(rpart.plot)
				library(RColorBrewer)
				fit <- rpart("""+target_variable_name+"""~., data = data1,xval = 10,cp=0,maxdepth = 4,minsplit=6)
				rpart.plot(fit,roundint=TRUE)
				}
				"""
	grdevices.jpeg(file="dec_tree.jpeg", width=1600, height=1600,quality=100)#
	if class_or_Reg =='Classification':
			rfunc=ro.r(rstring1)#
			p=rfunc(r_from_pd_df) 
			print("Class weights are ",priors)
	elif class_or_Reg =='Regression':
			rfunc=ro.r(rstring2)#
			p=rfunc(r_from_pd_df) 
	grdevices.dev_off()#
	from IPython.display import Image, display
	display(Image('dec_tree.jpeg'))




def Visualization(X, Y, class_or_Reg,LE):
	ohe = OneHotEncoder()
	cc = pd.DataFrame(X.select_dtypes('category')).astype(str)
	X_enc = ohe.fit_transform(cc)
	X_con = pd.get_dummies(X, columns = cc.columns)
	print("Categorical Columns considered:\n")
	print(cc.columns)
	print("Non-Categorical Columns considered:\n")
	print(X.drop(columns = cc.columns).columns)
	if class_or_Reg == 'Classification':
		from sklearn.tree import DecisionTreeClassifier
		from sklearn import tree
		import matplotlib.pyplot as plt
		from sklearn.preprocessing import LabelEncoder
# 		le = LabelEncoder()    #Y is already label encoded there's no need to label encode it twice.
# 		Y = le.fit_transform(Y)#encoding the target variable
		Yt=pd.DataFrame(Y)
		clf = DecisionTreeClassifier(max_depth = 5, min_samples_split=2,
									min_samples_leaf=0.01, random_state = 0,
									class_weight='balanced')
		clf.fit(X_con, Y)
		class_names=list(LE.inverse_transform(sorted(Yt[Yt.columns[0]].unique())))
		for i in range(len(class_names)):
			class_names[i]=str(class_names[i])
		print("value=[n1,n2,n3...] where n1,n2,n3 are the number of samples of the classes in the order     \nvalue="+str(LE.inverse_transform(sorted(Yt[Yt.columns[0]].unique()))))
		print(class_names)
		dot_data = tree.export_graphviz(clf,out_file=None,
										feature_names=X_con.columns,
										class_names=class_names,filled=True, impurity=False,
										proportion = True, rounded=True, special_characters=True)
		coX = list(zip(cc.columns, ohe.categories_))
		sx = list(cc.columns)
		new_dot = dot_data
		new_dot = re.sub('&le;', '&ge;', new_dot)
		for i, col in enumerate(sx):
			for cat in ohe.categories_[i]:
				new_dot = re.sub(f"{re.escape(col)}_{re.escape(cat)} &ge; 0.5", f"{col} = {cat}", new_dot)
		new_dot = re.sub('labelangle=45, headlabel="True"', 'labelangle=45, headlabel="False"', new_dot)
		new_dot = re.sub('labelangle=-45, headlabel="False"', 'labelangle=-45, headlabel="True"', new_dot)
		graph = pydotplus.graph_from_dot_data(new_dot)
		graph.write_png('Dtree.png')
	else:
		from sklearn.tree import DecisionTreeRegressor
		from sklearn import tree
		import matplotlib.pyplot as plt
		clf = DecisionTreeRegressor(max_depth = 5, min_samples_split=2, min_samples_leaf=0.01, random_state = 0)
		clf.fit(X_con, Y)
		dot_data = tree.export_graphviz(clf, out_file=None,
										feature_names=X_con.columns,
										filled=True, impurity=False, proportion = True, rounded=True,
										special_characters=True)
		coX = list(zip(cc.columns, ohe.categories_))
		sx = list(cc.columns)
		new_dot = dot_data
		new_dot = re.sub('&le;', '&ge;', new_dot)
		for i, col in enumerate(sx):
			for cat in ohe.categories_[i]:
				new_dot = re.sub(f"{re.escape(col)}_{re.escape(cat)} &ge; 0.5", f"{col} = {cat}", new_dot)
		new_dot = re.sub('labelangle=45, headlabel="True"', 'labelangle=45, headlabel="False"', new_dot)
		new_dot = re.sub('labelangle=-45, headlabel="False"', 'labelangle=-45, headlabel="True"', new_dot)
		graph = pydotplus.graph_from_dot_data(new_dot)
		graph.write_png('Dtree.png')
