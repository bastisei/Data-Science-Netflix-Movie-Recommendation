import pandas as pd
from surprise import Dataset, Reader, accuracy, SVD, NMF
from surprise.model_selection import cross_validate, train_test_split
import pickle

# Read data
df = pd.read_csv("../../data/raw/ratings_small.csv")

def validate_model(df, model_type='SVD'):
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    trainset, testset = train_test_split(data, test_size=0.25)

    if model_type == 'SVD':
        algo = SVD()
    elif model_type == 'NMF':
        algo = NMF()
    
    algo.fit(trainset)
    predictions = algo.test(testset)
    rmse = accuracy.rmse(predictions, verbose=True)

    results = cross_validate(algo, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)
    return algo

def save_model(model, filename):
    with open(filename, 'wb') as file:
        pickle.dump(model, file)
    print(f"Model saved to {filename}")

if __name__ == "__main__":
    # Train, validate, and save SVD model
    print("Evaluating and saving SVD Model:")
    svd_model = validate_model(df, 'SVD')
    save_model(svd_model, 'svd_model.pkl')

    # Train, validate, and save NMF model
    print()
    print("Evaluating and saving NMF Model:")
    nmf_model = validate_model(df, 'NMF')
    save_model(nmf_model, 'nmf_model.pkl')
