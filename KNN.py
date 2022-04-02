from collections import Counter
import numpy as np

def euclidean_distance(x1,x2):
	return np.sqrt(np.sum((x1-x2)**2))


class KNN:


	def  __init__(self,k=3):
		self.k=k


	def fit(self,X,y):
		self.X_train=X
		self.y_train=y


	def predict(self,X):
		y_pred= [self._predict(x) for x in X] #for all value of X
		return np.array(y_pred) #converts values into list


	def _predict(self,x):
		#compute the distance between x and all examples in training set
		distances= [euclidean_distance(x,x_train) for x_train in self.X_train]

		#sort the distances and return indices of the first k neighbors 
		k_idx= np.argsort(distances)[:self.k] #returns the total k values top 

		#extract the labels from the indices 
		k_neighbor_label= [self.y_train[i] for i in k_idx]
		#return most common found using most_common and counter function
		most_common= Counter(k_neighbor_label).most_common(1)
		return most_common[0][0]



if __name__== "__main__":
	from matplotlib.colors import ListedColormap
	from sklearn import datasets
	from sklearn.model_selection import train_test_split
	cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])


	def accuracy(y_true,y_pred):
		accuracy= np.sum(y_true==y_pred)/ len(y_true)
		return accuracy


	iris= datasets.load_iris()
	X,y= iris.data,iris.target

	X_train,X_test, y_train,y_test= train_test_split(
		X,y,test_size=0.2,random_state=1234)


	k=3
	clf= KNN(k=3)
	clf.fit(X_train,y_train)
	predictions= clf.predict(X_test)
	print("KNN Accuracy: ", accuracy(y_test,predictions))
