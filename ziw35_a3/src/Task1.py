import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import DecisionTreeRegressor
import sklearn.model_selection


class Task1:
    # please feel free to create new python files, adding functions and attributes to do training, validation, testing

    def __init__(self):
        print("================Task 1================")
        return

    def load_data(self, filename1, filename2):#load the train and test data as df1, df2
        self.df1 = pd.read_csv(filename1, header=None, delimiter='\t', na_values="n/a")
        self.df2 = pd.read_csv(filename2, header=None, delimiter='\t', na_values="n/a")

    def preprocess(self, classcolumnname):
        one_hot_training_predictors = pd.get_dummies(self.df1)
        self.x_train = one_hot_training_predictors.drop(classcolumnname, axis=1)
        self.y_train = one_hot_training_predictors[classcolumnname]
        self.df2['0_GP'] = int('0')  # Missing columns after encoding are added with values initialized to 0 in test dataset
        self.df2['9_health'] = int('0')
        self.df2['15_school'] = int('0')
        self.df2['15_school family paid'] = int('0')
        self.df2['15_school  paid'] = int('0')
        one_hot_testing_predictors = pd.get_dummies(self.df2)
        self.x_test = one_hot_testing_predictors.drop(classcolumnname, axis=1)
        self.y_test = one_hot_testing_predictors[classcolumnname]

    def trainlinearregression(self):
        self.lm = LinearRegression()
        self.lm.fit(self.x_train, self.y_train)
        lm_scores = sklearn.model_selection.cross_val_score(self.lm, self.x_train, self.y_train, cv=10).mean
        return lm_scores

    def traindecisiontreeregressor(self):
        regressor = DecisionTreeRegressor(random_state=0, criterion="mae")
        self.dt = regressor.fit(self.x_train, self.y_train)
        dt_scores = sklearn.model_selection.cross_val_score(self.dt, self.x_train, self.y_train, cv=10)
        return dt_scores

    def model_1_run(self):
        filename1 = 'data//assign3_students_train.txt'
        filename2 = 'data//assign3_students_test.txt'
        classcolumnname = 27
        self.load_data(filename1, filename2)
        self.preprocess(classcolumnname)
        # evaluate learned model on testing data
        lm = self.trainlinearregression()
        prediction = self.lm.predict(self.x_test)
        # The mean squared error
        print("Model 1: Mean squared error: %.2f"
              % mean_squared_error(self.y_test, prediction))
        return
        # Train the model 1 with your best hyper parameters (if have) and features on training data.
        # Evaluate learned model on testing data, and print the results

    def model_2_run(self):
        filename1 = 'data//assign3_students_train.txt'
        filename2 = 'data//assign3_students_test.txt'
        classColumnname = 27
        self.load_data(filename1, filename2)
        self.preprocess(classColumnname)
        # evaluate learned model on testing data
        dt = self.traindecisiontreeregressor()
        pred = self.dt.predict(self.x_test)
        #print(pred)
        # The mean squared error
        print("Model 2: Mean squared error: %.2f"
              % mean_squared_error(self.y_test, pred))
        return
        # Train the model 2 with your best hyper parameters (if have) and features on training data
        # Evaluate learned model on testing data, and print the results.


if __name__ == '__main__':
    task1 = Task1()
    task1.model_1_run()
    task1.model_2_run()
