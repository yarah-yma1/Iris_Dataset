import numpy as np
import pandas as pd
import seaborn as sns
sns.set_palette('husl')
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split  
from sklearn.linear_model import LogisticRegression  
from sklearn.tree import DecisionTreeClassifier 
from sklearn import svm  
from sklearn.neighbors import KNeighborsClassifier  
from sklearn import metrics  


# data class
class IrisData:
    def __init__(self, file='Iris.csv'):
        self.iris = pd.read_csv(file)
        self.iris1 = self.iris.drop('Id', axis=1)

    def info(self):
        print(self.iris.shape)
        print(self.iris.head())
        print(self.iris.tail())
        print(self.iris.info())
        print(self.iris.describe())
        print(self.iris['Species'].value_counts())

    def get_features_and_labels(self):
        X = self.iris.drop(['Id', 'Species'], axis=1)
        y = self.iris['Species']
        return X, y


# visual class
class IrisVisualizer:
    def __init__(self, iris1):
        self.iris1 = iris1

    def pairplot(self):
        g = sns.pairplot(self.iris1, hue='Species', markers='+')
        plt.show()

    def scatter_plots(self):
        # Sepal scatter
        fig = self.iris1[self.iris1.Species=='Iris-setosa'].plot.scatter(
            x='SepalLengthCm', y='SepalWidthCm', color='orange', label='Setosa')
        self.iris1[self.iris1.Species=='Iris-versicolor'].plot.scatter(
            x='SepalLengthCm', y='SepalWidthCm', color='blue', label='versicolor', ax=fig)
        self.iris1[self.iris1.Species=='Iris-virginica'].plot.scatter(
            x='SepalLengthCm', y='SepalWidthCm', color='green', label='virginica', ax=fig)
        fig.set_xlabel("Sepal Length")
        fig.set_ylabel("Sepal Width")
        fig.set_title("Sepal Length VS Width")
        plt.show()

        # Petal scatter
        fig = self.iris1[self.iris1.Species=='Iris-setosa'].plot.scatter(
            x='PetalLengthCm', y='PetalWidthCm', color='orange', label='Setosa')
        self.iris1[self.iris1.Species=='Iris-versicolor'].plot.scatter(
            x='PetalLengthCm', y='PetalWidthCm', color='blue', label='versicolor', ax=fig)
        self.iris1[self.iris1.Species=='Iris-virginica'].plot.scatter(
            x='PetalLengthCm', y='PetalWidthCm', color='green', label='virginica', ax=fig)
        fig.set_xlabel("Petal Length")
        fig.set_ylabel("Petal Width")
        fig.set_title("Petal Length VS Width")
        plt.show()

    def histograms(self):
        self.iris1.hist(edgecolor='black')
        plt.show()

    def violin_plots(self):
        for col in ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']:
            sns.violinplot(y='Species', x=col, data=self.iris1, inner='quartile')
            plt.show()

    def heatmap(self):
        plt.figure(figsize=(10,8)) 
        sns.heatmap(self.iris1.drop('Species', axis=1).corr(), annot=True, cmap='cubehelix_r')
        plt.show()


# model train
class IrisModel:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=10
        )

    def train_logistic_regression(self):
        model = LogisticRegression()
        model.fit(self.X_train, self.y_train)
        acc = metrics.accuracy_score(model.predict(self.X_test), self.y_test)
        print("Logistic Regression Accuracy:", acc)
        return acc

    def train_decision_tree(self):
        model = DecisionTreeClassifier()
        model.fit(self.X_train, self.y_train)
        acc = metrics.accuracy_score(model.predict(self.X_test), self.y_test)
        print("Decision Tree Accuracy:", acc)
        return acc

    def train_svm(self):
        model = svm.SVC()
        model.fit(self.X_train, self.y_train)
        acc = metrics.accuracy_score(model.predict(self.X_test), self.y_test)
        print("SVM Accuracy:", acc)
        return acc

    def train_knn(self, k=3):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(self.X_train, self.y_train)
        acc = metrics.accuracy_score(model.predict(self.X_test), self.y_test)
        print(f"KNN Accuracy (k={k}):", acc)
        return acc

    def knn_accuracy_vs_k(self):
        scores = []
        for i in range(1, 11):
            model = KNeighborsClassifier(n_neighbors=i)
            model.fit(self.X_train, self.y_train)
            acc = metrics.accuracy_score(model.predict(self.X_test), self.y_test)
            scores.append(acc)
        plt.plot(range(1, 11), scores)
        plt.xticks(range(1, 11))
        plt.title("KNN Accuracy vs K")
        plt.show()
        return scores


# evaluate class
class IrisEvaluator:
    def __init__(self, scores):
        self.scores = scores

    def summary(self):
        models = pd.DataFrame({
            'Model': ['Logistic Regression', 'Decision Tree', 'Support Vector Machines', 'K-Nearest Neighbours'],
            'Score': self.scores
        })
        print(models.sort_values(by='Score', ascending=False))


# main class
if __name__ == "__main__":
    # load data
    data = IrisData()
    data.info()

    # visualize
    viz = IrisVisualizer(data.iris1)
    viz.pairplot()
    viz.scatter_plots()
    viz.histograms()
    viz.violin_plots()
    viz.heatmap()

    # train models
    X, y = data.get_features_and_labels()
    model_trainer = IrisModel(X, y)

    acc_log = model_trainer.train_logistic_regression()
    acc_dt = model_trainer.train_decision_tree()
    acc_svm = model_trainer.train_svm()
    acc_knn = model_trainer.train_knn(k=3)

    model_trainer.knn_accuracy_vs_k()

    # evaluate
    evaluator = IrisEvaluator([acc_log, acc_dt, acc_svm, acc_knn])
    evaluator.summary()
