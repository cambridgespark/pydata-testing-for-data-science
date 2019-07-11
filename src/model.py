from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.transformers import CategoriesExtractor, CountryTransformer, GoalAdjustor, TimeTransformer


class KickstarterModel:

    # Update parameters here after re-tuning the model
    params = {"penalty": "l1", "C": 1.7, "solver": "liblinear"}

    def __init__(self):

        self.model = None
        self.preprocessor = None

    def preprocess_training_data(self, df):
        # Processor for categories with one-hot encoding
        cat_processor = Pipeline([("extractor", CategoriesExtractor()),
                                  ("one_hot",
                                   OneHotEncoder(sparse=False,
                                                 handle_unknown="ignore"))])

        # Processor for countries with one-hot encoding
        country_processor = Pipeline([("transfomer", CountryTransformer()),
                                      ("one_hot",
                                       OneHotEncoder(sparse=False,
                                                     handle_unknown="ignore"))])

        # First level of column specific transformations
        col_transformer = ColumnTransformer([
            ("goal", GoalAdjustor(), ["goal", "static_usd_rate"]),
            ("categories", cat_processor, ["category"]),
            ("disable_communication", "passthrough", ["disable_communication"]),
            ("time", TimeTransformer(),
             ["deadline", "created_at", "launched_at"]),
            ("countries", country_processor, ["country"])
        ])

        # Add a scaling stage
        self.preprocessor = Pipeline([("col_transformer", col_transformer),
                                      ("scaler", StandardScaler())])

        # Return X_train and y_train
        X_train = self.preprocessor.fit_transform(df.drop("state", axis=1))
        y_train = df.state.map({"failed": 0, "successful": 1})

        return X_train, y_train

    def fit(self, X, y):
        self.model = LogisticRegression(**self.params)
        self.model.fit(X, y)

    def preprocess_unseen_data(self, df):
        X_test = self.preprocessor.transform(df)
        return X_test

    def predict(self, X):

        return self.model.predict(X)
