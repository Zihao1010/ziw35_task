import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression as LR
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score,classification_report
from sklearn.neighbors import KNeighborsClassifier


class Task2:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing
    def __init__(self):
        print("================Task 2================")
        df1 = pd.read_csv('data//assign3_students_train.txt', header=None, delimiter='\t')
        df2 = pd.read_csv('data//assign3_students_test.txt', header=None, delimiter='\t')
        df0 = pd.concat([df1, df2], axis=0, ignore_index=True)
        data = df0.drop(8, axis=1)
        target = df0[8]
        x = pd.get_dummies(data)
        le = LabelEncoder()
        le.fit(["teacher", "health", "services", "at_home", "other"])
        y = le.transform(target)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        sc = StandardScaler()
        self.x_train_std = sc.fit_transform(self.x_train)
        self.x_test_std = sc.transform(self.x_test)
        return

    def print_category_results(self, category, precision, recall, f1):
        print("Category\t" + category + "\tF1\t" + str(f1) + "\tPrecision\t" + str(precision) + "\tRecall\t" + str(
            recall))

    def print_macro_results(self, accuracy, precision, recall, f1):
        print("Accuracy\t" + str(accuracy) + "\tMacro_F1\t" + str(f1) + "\tMacro_Precision\t" + str(
            precision) + "\tMacro_Recall\t" + str(recall))

    def model_1_run(self):
        print("Model 1:")
        lr1 = LR(C=3, class_weight=None, dual=False, fit_intercept=True,
                intercept_scaling=1, l1_ratio=None, max_iter=1000,
                multi_class='multinomial', n_jobs=None, penalty='l2',
                random_state=None, solver='lbfgs', tol=0.0001, verbose=0,
                warm_start=False)
        lr1.fit(self.x_train_std, self.y_train)
        y_pred = lr1.predict(self.x_test_std)
        confusion_matrix(y_true=self.y_test, y_pred=y_pred)
        target_names = ["teacher", "health", "service", "at_home", "other"]
        accuracy = '%.3f' % accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, target_names=target_names, output_dict=True)
        macro_precision = '%.3f' % report['macro avg']['precision']
        macro_recall = '%.3f' % report['macro avg']['recall']
        macro_f1 = '%.3f' % report['macro avg']['f1-score']
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        # Evaluate learned model on testing data, and print the results.
        self.print_macro_results(accuracy, macro_precision, macro_recall, macro_f1)
        categories = ["teacher", "health", "service", "at_home", "other"]
        for category in categories:
            precision = '%.3f' % report[category]['precision']
            recall = '%.3f' % report[category]['recall']
            f1 = '%.3f' % report[category]['f1-score']
            self.print_category_results(category, precision, recall, f1)
        return

    def model_2_run(self):
        print("--------------------\nModel 2:")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        knn = KNeighborsClassifier(algorithm='auto', leaf_size=30, metric='minkowski',
                                   metric_params=None, n_jobs=None, n_neighbors=36, p=2,
                                   weights='uniform')
        knn.fit(self.x_train_std, self.y_train)
        y_pred = knn.predict(self.x_test_std)
        confusion_matrix(y_true=self.y_test, y_pred=y_pred)
        target_names = ["teacher", "health", "service", "at_home", "other"]
        accuracy = '%.3f' % accuracy_score(self.y_test, y_pred)
        report = classification_report(self.y_test, y_pred, target_names=target_names, output_dict=True)
        macro_precision = '%.3f' % report['macro avg']['precision']
        macro_recall = '%.3f' % report['macro avg']['recall']
        macro_f1 = '%.3f' % report['macro avg']['f1-score']
        # Evaluate learned model on testing data, and print the results.
        self.print_macro_results(accuracy, macro_precision, macro_recall, macro_f1)
        categories = ["teacher", "health", "service", "at_home", "other"]
        for category in categories:
            precision = '%.3f' % report[category]['precision']
            recall = '%.3f' % report[category]['recall']
            f1 = '%.3f' % report[category]['f1-score']
            self.print_category_results(category, precision, recall, f1)
        return


if __name__ == '__main__':
    task2 = Task2()
    task2.model_1_run()
    task2.model_2_run()