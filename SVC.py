import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm, metrics
from imblearn.over_sampling import SMOTE

def runSVCrbf(X, y, X_validation):
    smote_2 = SMOTE(sampling_strategy=.3)
    smote_3 = SMOTE(sampling_strategy=.4)
    smote_4 = SMOTE(sampling_strategy=.5)
    X_2, y_2 = smote_2.fit_resample(X, y)
    X_3, y_3 = smote_3.fit_resample(X, y)
    X_4, y_4 = smote_4.fit_resample(X, y)
    arr_X = [X, X_2, X_3, X_4]
    arr_y = [y, y_2, y_3, y_4]
    for i in range(len(arr_y)):
        print("----DATASET------", i)
        X_train, X_test, y_train, y_test = train_test_split(np.asarray(arr_X[i]), np.ravel(np.asarray(arr_y[i])),
                                                            test_size=0.4)
        for j in range(1, 5):
            print()
            print()
            print("C: ", j)
            clf = svm.SVC(kernel='rbf', gamma=0.5, C=j)
            clf.fit(X_train, y_train)
            # make predictions
            expected = y_test
            predicted = np.ravel(clf.predict(X_test))
            runDiagnostics(expected, predicted)
            kaggle_test = clf.predict(X_validation)
            print("222 patients and num of no show is ", sum(kaggle_test))


def runSVCLinear(X, y, X_validation):
    smote_2 = SMOTE(sampling_strategy=.3)
    smote_3 = SMOTE(sampling_strategy=.4)
    smote_4 = SMOTE(sampling_strategy=.5)
    X_2, y_2 = smote_2.fit_resample(X, y)
    X_3, y_3 = smote_3.fit_resample(X, y)
    X_4, y_4 = smote_4.fit_resample(X, y)
    arr_X = [X, X_2, X_3, X_4]
    arr_y = [y, y_2, y_3, y_4]
    for i in range(len(arr_y)):
        print("----DATASET------", i)
        X_train, X_test, y_train, y_test = train_test_split(np.asarray(arr_X[i]), np.ravel(np.asarray(arr_y[i])),
                                                            test_size=0.5)
        for j in range(1, 5):
            print()
            print()
            print("C: ", j)
            svm_linear = svm.LinearSVC(penalty="l2", loss="hinge", C=j)
            svm_linear.fit(X_train, y_train)
            # make predictions
            expected = y_test
            predicted = np.ravel(svm_linear.predict(X_test))
            runDiagnostics(expected, predicted)
            print("coeff: ", svm_linear.coef_)
            kaggle_test = svm_linear.predict(X_validation)
            print("222 patients and num of no show is ", sum(kaggle_test))

def runDiagnostics(expected, predicted):
    # summarize the fit of the model
    print(metrics.classification_report(expected, predicted))
    confusion_matrix = metrics.confusion_matrix(expected, predicted)
    print(confusion_matrix)
    balanced_accuracy = (0.5 * (confusion_matrix[0][0] / (confusion_matrix[0][1] * 1.0 + confusion_matrix[0][0])
                                + confusion_matrix[1][1] / (
                                    confusion_matrix[1][1] * 1.0 + confusion_matrix[1][0])))
    print("balanced accuracy: ", balanced_accuracy)

def main():
    print("norm")
    X = np.load('xTrain_norm')
    X_val = np.load('xVal_norm')
    runSVCrbf(X, y, X_val)
    runSVCLinear(X, y, X_val)
    print("MinMax")
    X = np.load('xTrain_MinMax')
    X_val = np.load('xVal_MinMax')
    runSVCrbf(X, y, X_val)
    runSVCLinear(X, y, X_val)


if __name__ == '__main__':
    main()