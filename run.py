import argparse
import os
import joblib
import urllib.request

import pandas as pd
import sys

import pytest

from src.model import KickstarterModel as Model

DATASET_URL = "https://s3-eu-west-1.amazonaws.com/kate-datasets/kickstarter/"
TRAIN_NAME = "train.zip"
TEST_NAME = "test.zip"

DATA_DIR = "data"
JOBLIB_NAME = 'model.joblib'


def setup_data():
    if not os.path.isdir(DATA_DIR):
        os.makedirs(DATA_DIR)

    for filename in [TRAIN_NAME, TEST_NAME]:
        print("Downloading {}...".format(filename))
        req = urllib.request.urlopen(DATASET_URL + filename)
        data = req.read()

        with open(os.path.join(DATA_DIR, filename), "wb") as f:
            f.write(data)


def train_model():
    df = pd.read_csv(os.sep.join([DATA_DIR, TRAIN_NAME]))

    my_model = Model()
    X_train, y_train = my_model.preprocess_training_data(df)
    my_model.fit(X_train, y_train)

    # Save to pickle
    with open(JOBLIB_NAME, 'wb') as f:
        joblib.dump(my_model, f)


def test_model():
    df = pd.read_csv(os.sep.join([DATA_DIR, TEST_NAME]))

    # Load JOB
    with open(JOBLIB_NAME, 'rb') as f:
        my_model = joblib.load(f)

    X_test = my_model.preprocess_unseen_data(df)
    preds = my_model.predict(X_test)
    print("### Your predictions ###")
    print(preds)


def main():
    parser = argparse.ArgumentParser(
        description="A command line-tool to manage the project.")
    parser.add_argument(
        'stage',
        metavar='stage',
        type=str,
        choices=['setup', 'train', 'test', 'unittest', 'coverage', 'hypothesis', 'exercises'],
        help="Stage to run. Either setup, train, test, unittest, coverage, hypothesis or exercises")

    if len(sys.argv[1:]) == 0:
        parser.print_help()
        parser.exit()

    stage = parser.parse_args().stage

    if stage == "setup":
        setup_data()
        print("\nSetup was successful!")

    elif stage == "train":
        print("Training model...")
        train_model()

    elif stage == "test":
        print("Testing model...")
        test_model()

    elif stage == "unittest":
        print("Unittesting model...")
        pytest.main(['-v', 'tests'])

    elif stage == "coverage":
        print("Running coverage...")
        pytest.main(['--cov-report', 'term-missing', '--cov=src/', 'tests/'])

    elif stage == "hypothesis":
        print("Running hypothesis...")
        pytest.main(['-v', '--hypothesis-show-statistics', 'tests/test_transformers_hypothesis.py'])

    elif stage == "exercises":
        print("Running the exercises...")
        pytest.main(['-v', 'exercises'])

if __name__ == "__main__":
    main()
