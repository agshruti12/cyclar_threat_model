import json
from src.features import compute_track_features

def load_tracks(path: str):
    with open(path, "r") as f:
        return json.load(f)

def save_track_features(features_list, path: str):
    import os
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(features_list, f)

if __name__ == "__main__":
    tracks = load_tracks("data/processed/sample_bike_ride_tracks.json")
    features_per_track = []
    for t in tracks:
        ft = compute_track_features(t)
        features_per_track.append(ft)
    save_track_features(features_per_track, "data/processed/sample_bike_ride_tracks_features.json")
    print(f"Saved features for {len(features_per_track)} tracks")
