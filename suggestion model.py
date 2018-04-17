#Ongoing framework for recommendation system using KNN (K nearest neighbor)



import numpy as np
import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#Here we put our CSV (comma seperated file - our data set)

#training set
train = pd.read_csv('nameoffile.train.csv')
#test set
test = pd.read_csv('nameoffile.test.csv')



# create design matrix X and target vector y
X = np.array(df.ix[:, 0:4])
y = np.array(df['SuggestedCategories'])


# split into train and test we set it to 15% it is usually a 80/20 to 70/30 split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15)

# instantiate K (k = 3)
knn = KNeighborsClassifier(n_neighbors=3)

# fitting the model
knn.fit(X_train, y_train)

# predict the response
y_predict = knn.predict(X_test)

# evaluate accuracy

print("Accuracy of 1NN: %.2f %%" %(100*accuracy_score(y_test, y_predict)))




gender = ["maleX", "femaleXY"]

Categories = ["clothes", "jewlery", "home", "decor", "food", "kids", "education", "holiday", "festivity"]

subcatclothesX = ["summerX", "casualX", "formalX", "beachwearX","outerwearX"]

subcatclothesXY = ["summerXY", "casualXY", "occasionsXY", "formalXY", "outerwearXY"]


Seasonal = ["winter", "spring", "summer", "fall"]

class SuggestedCategories
    def __init__ (self, clothes, jewlery, home, decore, food, kids, education, holiday, festivities):
        self.clothes = clh
        self.jewlery = jwl
        self.home = hm
        self.decore = dc
        self.food = fd
        self.kids = kd
        self.education = edu
        self.holiday = hl
        self.festivities = fst




        pass


class ViewedCategory:
    def __init__ (self, ViewedCategory, shownitems):
        self.viewedCategory = VC
        self.shownitems = SI

        VC + categories = SI

clickedItems == SuggestedItems == ShownItems

return (ViewedCategory)


