
from sklearn.model_selection import train_test_split
from sklearn import model_selection
import pandas as pd

def get_project_data():

    chunksize = 100000
    tfr = pd.read_csv('handwritten_digits_images.csv', chunksize=chunksize, iterator=True)
    X_raw = pd.concat(tfr, ignore_index=True)
    X = X_raw.reshape(X_raw.shape[0], 28, 28)

    y = pd.read_csv("handwritten_digits_labels.csv")
    print(X)

    seed = 4233
    # Shuffle and split the data into train and a concatenation of validation and test sets with a ratio of 0.5/0.5:
    X_train, X_val_test, y_train, y_val_test = model_selection.train_test_split(
        X, y, test_size=0.5, shuffle=True, random_state=seed
    )
    seed = 4555
    # Shuffle and split the data into validation and test sets with a ratio of 0.5/0.5:
    X_val, X_test, y_val, y_test = model_selection.train_test_split(
        X_val_test, y_val_test, test_size=0.5, shuffle=True, random_state=seed
    )

    return (X_train, y_train, X_val, y_val, X_test, y_test)


def main():
    print(get_project_data())
    None

if __name__ == "__main__":
    main()