import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, hamming_loss
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier


class Task3:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 3================")
        df1 = pd.read_csv('data//assign3_students_train.txt', header=None, delimiter='\t')
        df2 = pd.read_csv('data//assign3_students_test.txt', header=None, delimiter='\t')
        df0 = pd.concat([df1, df2], axis=0, ignore_index=True)
        a = pd.get_dummies(df0[15])
        for i in a.index:
            if a.loc[i, 'family paid'] == 1:
                a.loc[i, 'family'] = 1
                a.loc[i, 'paid'] = 1
            elif a.loc[i, 'school family'] == 1:
                a.loc[i, 'family'] = 1
                a.loc[i, 'school'] = 1
            elif a.loc[i, 'school paid'] == 1:
                a.loc[i, 'paid'] = 1
                a.loc[i, 'school'] = 1
            elif a.loc[i, 'school family paid'] == 1:
                a.loc[i, 'paid'] = 1
                a.loc[i, 'school'] = 1
                a.loc[i, 'family'] = 1
            else:
                continue
        y = a[['family', 'paid', 'school', 'no']]
        y = y.values
        data = df0.drop(15, axis=1)
        x = pd.get_dummies(data)
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(x, y, test_size=0.2, random_state=0)
        sc = StandardScaler()
        self.x_train_std = sc.fit_transform(self.x_train)
        self.x_test_std = sc.transform(self.x_test)
        return

    def model_1_run(self):
        print("Model 1:")
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        # Evaluate learned model on testing data, and print the results.
        lr = LogisticRegression(C=1, class_weight=None,
                                dual=False, fit_intercept=True,
                                intercept_scaling=1,
                                l1_ratio=None, max_iter=100,
                                multi_class='ovr',
                                n_jobs=None, penalty='l2',
                                random_state=None,
                                solver='liblinear', tol=0.0001,
                                verbose=0, warm_start=False)
        ovs = OneVsRestClassifier(lr)
        ovs.fit(self.x_train_std, self.y_train)
        y_pred = ovs.predict(self.x_test_std)
        accuracy = '%.3f' % accuracy_score(self.y_test, y_pred)
        hl = '%.3f' % hamming_loss(self.y_test, y_pred)

        # Evaluate learned model on testing data, and print the results.
        print("Accuracy\t" + str(accuracy) + "\tHamming loss\t" + str(hl))
        return

    def model_2_run(self):
        print("--------------------\nModel 2:")
        # Train the model 2 with your best hyper parameters (if have) and features on training data.
        # Evaluate learned model on testing data, and print the results.
        knn = KNeighborsClassifier(algorithm='auto',
                                   leaf_size=30,
                                   metric='minkowski',
                                   metric_params=None,
                                   n_jobs=None, n_neighbors=45,
                                   p=2, weights='uniform'
                                   )
        ovs = OneVsRestClassifier(knn)
        ovs.fit(self.x_train_std, self.y_train)  # train model
        y_pred = ovs.predict(self.x_test_std)  # prediction
        accuracy = '%.3f' % accuracy_score(self.y_test, y_pred)
        hl = '%.3f' % hamming_loss(self.y_test, y_pred)
        # Evaluate learned model on testing data, and print the results.
        print("Accuracy\t" + str(accuracy) + "\tHamming loss\t" + str(hl))
        return


if __name__ == '__main__':
    task3 = Task3()
    task3.model_1_run()
    task3.model_2_run()