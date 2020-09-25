import pydotplus
from sklearn.preprocessing import OneHotEncoder
import pandas as pd
import numpy as np
import re
def Visualization(X, Y, class_or_Reg):
	ohe = OneHotEncoder()
	cc = pd.DataFrame(X.select_dtypes('category')).astype(str)
	X_enc = ohe.fit_transform(cc)
	X_con = pd.get_dummies(X, columns = cc.columns)
	if class_or_Reg == 'Classification':
		from sklearn.tree import DecisionTreeClassifier
		from sklearn import tree
		import matplotlib.pyplot as plt
		from sklearn.preprocessing import LabelEncoder
		le = LabelEncoder()
		Y = le.fit_transform(Y)#encoding the target variable
		Yt=pd.DataFrame(Y)
		clf = DecisionTreeClassifier(max_depth = 5, min_samples_split=2,
									min_samples_leaf=0.01, random_state = 0,
									class_weight='balanced')
		clf.fit(X_con, Y)
		class_names=list(le.inverse_transform(sorted(Yt[Yt.columns[0]].unique())))
		for i in range(len(class_names)):
			class_names[i]=str(class_names[i])
		print("value=[n1,n2,n3...] where n1,n2,n3 are the number of samples of the classes in the order     \nvalue="+str(le.inverse_transform(sorted(Yt[Yt.columns[0]].unique()))))
		print(class_names)
		dot_data = tree.export_graphviz(clf,out_file=None,
										feature_names=X_con.columns,
										class_names=class_names,filled=True, impurity=False,
										proportion = True, rounded=True, special_characters=True)
		coX = list(zip(cc.columns, ohe.categories_))
		sx = list(cc.columns)
		new_dot = dot_data
		for i, col in enumerate(sx):
			for cat in ohe.categories_[i]:
				new_dot = re.sub(f"{re.escape(col)}_{re.escape(cat)} &le; 0.5", f"{col} &ne; {cat}", new_dot)
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
		for i, col in enumerate(sx):
			for cat in ohe.categories_[i]:
				new_dot = re.sub(f"{re.escape(col)}_{re.escape(cat)} &le; 0.5", f"{col} &ne; {cat}", new_dot)
		graph = pydotplus.graph_from_dot_data(new_dot)
		graph.write_png('Dtree.png')
