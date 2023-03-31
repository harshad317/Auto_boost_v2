import dependencies

class Preprocessing:
    def __init__(self, df, target, task = 'classification'):
        self.df = df
        self.task = task
        self.target = target
    def find_null_values(self, target):
        print('Looking for null values...')
        all_missing_values = 0
        for col in dependencies.tqdm(self.df.columns): 
            if self.df[f'{col}'].isnull().sum() > 0 and self.df[f'{col}'] is not self.target:
                all_missing_values += self.df[f'{col}'].isnull().sum()
                return f'{col} has {self.df[{col}].isnull().sum()} missing values.'
            else:
                return f'{col} has {self.df[{col}].isnull().sum()} missing values.'
        if all_missing_values == 0:
            print('There are no missing values in the dataset.')
        return self.df
    def fill_missing_values(self):
        print('Filling missing values...')
        for col in dependencies.tqdm(self.df.columns):
            if self.df[f'{col}'].dtype == 'O' and self.df[f'{col}'].isnull().sum() <= (dependencies.np.round(len(self.df[f'{col}'].nunique()), 0))/4:
                self.df[f'{col}'].fillna(self.df[f'{col}'].mode()[0], axis=0, inplace=True)
            elif self.df[f'{col}'].dtype == 'O' and self.df[f'{col}'].isnull().sum() > (dependencies.np.round(len(self.df[f'{col}'].nunique()), 0))/4:
                self.df = self.df.drop([f'{col}'], axis = 1)
            elif self.df[f'{col}'].dtype != 'O' and self.df[f'{col}'].isnull().sum() <= (dependencies.np.round(len(self.df[f'{col}'].nunique()), 0))/4:
                self.df[f'{col}'].fillna(self.df[f'{col}'].median(), axis=0, inplace=True)
            elif self.df[f'{col}'].dtype != 'O' and self.df[f'{col}'].isnull().sum() > (dependencies.np.round(len(self.df[f'{col}'].nunique()), 0))/4:
                self.df = self.df.drop([f'{col}'], axis= 1)
        return self.df
        print('Done with missing values...')
    def find_outliers(self):
        print('Looking for outliers...')
        for col in dependencies.tqdm([var for var in self.df.columns if self.df[f'{col}'].dtypes != 'O']):
            Q3 = dependencies.np.percentile(self.df[f'{col}'], 75, interpolation = 'midpoint')
            Q1 = dependencies.np.percentile(self.df[f'{col}'], 25, interpolation='midpoint')
            IQR = Q3 - Q1
            upper = Q3 + (1.5 * IQR)
            lower = Q1 - (1.5 * IQR)
            upper_array = dependencies.np.array(self.df[f'{col}'] >= upper)
            lower_array = dependencies.np.array(self.df[f'{col}'] <= lower)
            self.df[f'{col}'].drop(upper_array[0], inplace=True)
            self.df[f'{col}'].drop(lower_array[0], inplace=True)
        return self.df
        print('Done with outlier...')
    def categorical_column(self):
        for col in dependencies.tqdm([var for var in self.df.columns if self.df[f'{var}'].dtypes == 'O']):
            if self.df[f'{col}'].nunique() == 1:
                self.df = self.df.drop([f'{col}'], axis= 1)
            elif self.df[f'{col}'].nunique() <= 15:
                    dummy_cols = dependencies.pd.get_dummies(f'{col}', drop_first=True, prefix=f'{col}')
                    self.df = self.df.drop(f'{col}', axis= 1)
                    self.df = dependencies.pd.concat([self.df, dummy_cols], axis=1)
            elif self.df[f'{col}'].nunique() > 15 and self.df[f'{col}'].nunique() < len(self.df):
                le = dependencies.LabelEncoder()
                le.fit(self.df[f'{col}'])
                self.df[f'{col}'] = le.transform(self.df[f'{col}'])
        print('Done with categorical data...')
        return self.df
    def train_and_test(self):
        if self.task == 'classification':
            train, validation = dependencies.train_test_split(self.df, test_size= 0.25, random_state=2023, stratify= self.target)
        elif self.task == 'regression':
            train, validation = dependencies.train_test_split(self.df, test_size= 0.25, random_state= 2023)
        return train, validation
    def final_process(self, train, validation):
        cols = [var for var in train.columns if train[var] != self.target]
        X_train = train[cols]
        X_test = validation[cols]
        y_train = train[self.target]
        y_test = validation[self.target]
        return X_train, X_test, y_train, y_test