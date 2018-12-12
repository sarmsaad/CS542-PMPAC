import numpy as np
import graphviz
import sklearn

from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split

# SMOTE
# ratio majority/minority
def runDecisionTree(X, y, X_validation):
    smote_1 = SMOTE(sampling_strategy=.3)
    smote_2 = SMOTE(sampling_strategy=.4)
    smote_3 = SMOTE(sampling_strategy=.5)
    smote_4 = SMOTE(sampling_strategy=.6)
    X_1, y_1 = smote_1.fit_resample(X, y)
    X_2, y_2 = smote_2.fit_resample(X, y)
    X_3, y_3 = smote_3.fit_resample(X, y)
    X_4, y_4 = smote_4.fit_resample(X, y)
    arr_X = [X_1, X_2, X_3, X_4]
    arr_y = [y_1, y_2, y_3, y_4]
    tree = DecisionTreeClassifier()
    for i in range(len(arr_y)):
        print()
        print()
        print("PREDICTION: ", i)
        X_train, X_test, y_train, y_test = train_test_split(np.asarray(arr_X[i]), np.ravel(np.asarray(arr_y[i])), test_size=0.3)
        tree.fit(X_train, y_train)
        # make predictions
        expected = y_test
        predicted = np.ravel(tree.predict(X_test))
        # summarize the fit of the model
        print(metrics.classification_report(expected, predicted))
        confusion_matrix = metrics.confusion_matrix(expected, predicted)
        print(confusion_matrix)
        balanced_accuracy = (0.5 * (confusion_matrix[0][0]/(confusion_matrix[0][1] * 1.0 + confusion_matrix[0][0])
                             + confusion_matrix[1][1]/(confusion_matrix[1][1] * 1.0 + confusion_matrix[1][0])))
        print("balanced accuracy: ", balanced_accuracy)
        print("feature importance: ", tree.feature_importances_)
        kaggle_test = tree.predict(X_validation)
        print("222 patients and num of no show is ", sum(kaggle_test))

    ## .6 is best
    X_train, X_test, y_train, y_test = train_test_split(np.asarray(arr_X[3]), np.ravel(np.asarray(arr_y[3])), test_size=0.4)
    tree.fit(X_train, y_train)
    return tree

def visualizeTree(tree):
    feature_names = ['EMAIL', 'SMS', 'User_clicked_UI_button', 'Node_Viewed', 'User_completed_module', 'Patient_Rescheduled_By_Tenant',
    'Patient_Cancelled_By_Tenant', 'Added_Copilot', 'User_unsubscribed', 'age', 'regit_time']

    dot_data = sklearn.tree.export_graphviz(tree, out_file=None,
                          feature_names=feature_names,
                          filled=True, rounded=True,
                          special_characters=True,
                          max_depth=3)
    graph = graphviz.Source(dot_data)

    graph.render("outputTree")

def main():
    X = np.load('xTrain.npy')
    y = np.load('yTrain.npy')
    X_val = np.load('xVal.npy')
    final_tree = runDecisionTree(X, y, X_val)
    visualizeTree(final_tree)


if __name__ == '__main__':
    main()
