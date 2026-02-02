import json
import requests
from pathlib import Path

import config


def fetch_reddit_top_posts(subreddit: str, limit: int = 50) -> dict:
    """
    Fetch top posts from a specific subreddit using the public JSON endpoint.
    """
    # Use the example endpoint structure
    url = f"https://www.reddit.com/r/{subreddit}/top.json"
    params = {"limit": limit}

    # A custom User-Agent is mandatory to avoid being blocked
    headers = {"User-Agent": "DataScienceCourse/1.0"}

    r = requests.get(url, params=params, headers=headers, timeout=30)
    r.raise_for_status()

    return r.json()


def main() -> None:
    # Configuration based on task requirements
    subreddit_name = getattr(config, "SUBREDDIT", "datascience")
    post_limit = int(getattr(config, "REDDIT_LIMIT", 50))

    # Define the output path as specified: data/raw_source2.json
    root = Path(__file__).resolve().parent
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    out_path = data_dir / "raw_source2.json"

    try:
        print(f"Fetching top {post_limit} posts from r/{subreddit_name}...")
        raw_data = fetch_reddit_top_posts(subreddit_name, post_limit)

        # Save the raw JSON data to the specified file
        out_path.write_text(
            json.dumps(raw_data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Success: Data saved to {out_path}")

    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from Reddit: {e}")


if __name__ == "__main__":
    main()
