import dependencies
#import preprocessing
import feature_selection

X_train, X_test = feature_selection.feature_selection()

class modeling:
    def __init__(self, algorithm, metric, folds, X_train, X_test, y_train, y_test, task= 'classification'):
        self.algorithm = algorithm
        self.metric = metric
        self.folds = folds
        self.X_train = X_train
        self.X_test = X_test
        self.task  = task
    def creating_model(self):
        if self.task == 'classification':
            skf = dependencies.StratifiedKFold(n_splits=self.folds, shuffle=True, random_state=2023)
            score = []
            for fold, (train_idx, test_idx) in enumerate(self.X_train, self.y_train):
                print('=' * 15, 'Fold No:', fold+1, '=' * 15)
                X_tr, X_tst = self.X_train.iloc[train_idx], self.X_train.iloc[test_idx]
                y_tr, y_tst = self.y_train.iloc[train_idx], self.y_train.iloc[test_idx]
                model = self.algorithm
                model.fit(X_tr, y_tr, eval_set = [X_tst, y_tst], early_stopping_rounds = 50, verbose = 100)
                predict = model.predict(X_tst)
                predict_probability = model.predict_proba(X_tst)
                if self.metric == 'precision':
                    print('Precision score: ', dependencies.precision_score(y_tst, predict))
                    score.append(dependencies.precision_score(y_tst, predict))
                    final_predict = model.predict(self.X_test)
                elif self.metric == 'recall':
                    print('Recall score: ', dependencies.recall_score(y_tst, predict))
                    score.append(dependencies.recall_score(y_tst, predict))
                    final_predict = model.predict(self.X_test)
                elif self.metric == 'f1':
                    print('F1 score:', dependencies.f1_score(y_tst, predict))
                    score.append(dependencies.f1_score(y_tst, predict))
                    final_predict = model.predict(self.X_test)
                elif self.metric == 'f1_micro':
                    print('F1 score:', dependencies.f1_score(y_tst, predict, average = 'micro'))
                    score.append(dependencies.f1_score(y_tst, predict, average = 'micro'))
                    final_predict = model.predict(self.X_test)
                elif self.metric == 'f1_macro':
                    print('F1 score:', dependencies.f1_score(y_tst, predict, average = 'macro'))
                    score.append(dependencies.f1_score(y_tst, predict, average = 'macro'))
                    final_predict = model.predict(self.X_test)
                elif self.metric == 'f1_weighted':
                    print('F1 score:', dependencies.f1_score(y_tst, predict, average='weighted'))
                    score.append(dependencies.f1_score(y_tst, predict, average='weighted'))
                    final_predict = model.predict(self.X_test)
                elif self.metric == 'accuracy':
                    print('Accuracy: ', dependencies.accuracy_score(y_tst, predict))
                    score.append(dependencies.accuracy_score(y_tst, predict))
                    final_predict = model.predict(self.X_test)
                elif self.metric == 'log_loss':
                    print('Log loss score: ', dependencies.log_loss(y_tst, predict_probability))
                    score.append(dependencies.log_loss(y_tst, predict_probability))
                    final_predict = model.predict(self.X_test)
                elif self.metric == 'roc_auc':
                    print('ROC_AUC: ', dependencies.roc_auc_score(y_tst, predict_probability[:,1]))
                    score.append(dependencies.roc_auc_score(y_tst, predict_probability[:,1]))
                    final_predict = model.predict(self.X_test)
            print('Final score: ', score)
        if self.task == 'regression':
            skf = dependencies.KFold(n_splits=self.folds, shuffle=True, random_state=2023)
            score = []
            for fold, (train_idx, test_idx) in enumerate(self.X_train, self.y_train):
                print('=' * 15, 'Fold no:', fold + 1, '=' * 15)
                X_tr, X_tst = self.X_train.iloc[train_idx], self.X_train.iloc[test_idx]
                y_tr, y_tst = self.y_train.iloc[train_idx], self.y_train.iloc[test_idx]
                model = self.algorithm
                model.fit(X_tr, y_tr, eval_set = [X_tst, y_tst], early_stopping_rounds = 50, verbose = 500)
                predict = model.predict(X_tst)
                if self.metric == 'mse':
                    print('MSE: ', dependencies.mean_squared_error(y_tst, predict))
                    score.append(dependencies.mean_squared_error(y_tst, predict))
                    final_predict = model.predict(self.X_test)
                elif self.metric == 'rmse':
                    print('RMSE: ', dependencies.mean_squared_error(y_tst, predict, squared= False))
                    score.append(dependencies.mean_squared_error(y_tst, predict, squared= False))
                    final_predict = model.predict(self.X_test)
                elif self.metric == 'rmsle':
                    print('RMSLE: ', dependencies.mean_squared_log_error(y_tst, predict, squared= False))
                    score.append(dependencies.mean_squared_log_error(y_tst, predict, squared= False))
                    final_predict = model.predict(self.X_test)
                elif self.metric == 'mae':
                    print('MAE: ', dependencies.mean_absolute_error(y_tst, predict) )
                    score.append(dependencies.mean_absolute_error(y_tst, predict))
                    final_predict = model.predict(self.X_test)
            print('Final score: ', score)
        print('#' * 25)
        # Plot feature importance
        feature_importance = model.feature_importances_
        # make importances relative to max importance
        feature_importance = 100.0 * (feature_importance / feature_importance.max())
        sorted_idx = dependencies.np.argsort(feature_importance)
        pos = dependencies.np.arange(sorted_idx.shape[0]) + .5
        #plt.subplot(1, 1, 1)
        dependencies.plt.figure(figsize=(10,10))
        dependencies.plt.barh(pos, feature_importance[sorted_idx], align='center')
        dependencies.plt.yticks(pos, train.columns[sorted_idx])#boston.feature_names[sorted_idx])
        dependencies.plt.xlabel('Relative Importance')
        dependencies.plt.title('Variable Importance')
        dependencies.plt.show()
        print('#' * 25)
        return final_predict