import dependencies
import preprocessing

X_train, X_test, y_train, y_test = preprocessing.Preprocessing()

class feature_selection:
    def __init__(self, X_train, y_train, X_test, algorithm):
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.model = algorithm
    def selection(self):
        feat_selector = dependencies.BorutaPy(self.model, n_estimators = 'auto', verbose= 2, random_state=2023)
        feat_selector.fit(self.X_train, self.y_train)
        # Check selected features
        print(feat_selector.support_)
        # Select the chosen features from our dataframe.
        selected = self.X_train[:, feat_selector.support_]
        print ("")
        print ("Selected Feature Matrix Shape")
        print (selected.shape)
        print(feat_selector.ranking_)
        X_train_filtered = feat_selector.transform(self.X_train)
        X_test_filtered = feat_selector.transform(self.X_test)
        return X_train_filtered, X_test_filtered