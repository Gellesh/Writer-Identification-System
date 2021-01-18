from sklearn import svm
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB 
from sklearn.neighbors import KNeighborsClassifier 

""" Contains different types of classifiers """


def train_using_svm(features,labels):
    clf = svm.SVC(kernel='linear'  ,C=1.0) # Linear Kernel
    clf.fit(features, labels)
    return clf

def naive_Bayes(features,labels):
    gnb = GaussianNB().fit(features, labels) 
    return gnb
 
def KNN(features,labels):
    knn = KNeighborsClassifier(n_neighbors = 3).fit(features, labels) 
    return knn