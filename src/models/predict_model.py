import pickle
from surprise import Dataset, Reader
import pandas as pd

def load_model(filename):
    """Load a trained model from a file."""
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    return model

def generate_recommendations(user_id, algo, df):
    """Generate and print top 10 movie recommendations for a user."""
    reader = Reader(rating_scale=(0.5, 5))
    data = Dataset.load_from_df(df[['userId', 'movieId', 'rating']], reader)
    trainset = data.build_full_trainset()
    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)

    user_predictions = [pred for pred in predictions if pred.uid == user_id]
    user_predictions.sort(key=lambda x: x.est, reverse=True)
    top_10_recommendations = user_predictions[:10]

    print(f"Top 10 recommendations for User {user_id}:")
    for pred in top_10_recommendations:
        print(f"Movie ID: {pred.iid}, Estimated Rating: {pred.est:.2f}")

def main():
    df = pd.read_csv("../../data/raw/ratings_small.csv")
    svd_model = load_model('svd_model.pkl')
    nmf_model = load_model('nmf_model.pkl')
    user_id = 1
    generate_recommendations(user_id, svd_model, df)
    print()
    generate_recommendations(user_id, nmf_model, df)

if __name__ == "__main__":
    main()
