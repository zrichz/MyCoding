import requests
import time
import json

BASE_URL = "https://danbooru.donmai.us/tags.json"

def fetch_tags(limit=1000, sleep=1.0, max_tags=20000):
    """
    Fetches Danbooru tags with pagination.
    Returns a list of tag dictionaries.
    """
    page = 1
    all_tags = []

    while len(all_tags) < max_tags:
        params = {
            "limit": limit,
            "page": page,
            "search[order]": "count"
        }

        print(f"Fetching page {page}... (collected {len(all_tags)}/{max_tags})")
        response = requests.get(BASE_URL, params=params)

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            break

        data = response.json()
        if not data:
            break

        all_tags.extend(data)
        page += 1
        time.sleep(sleep)

    return all_tags[:max_tags]


def filter_and_group_tags(tags):
    """
    Filters out deprecated tags and groups by category.
    Returns dict of category -> list of (name, post_count) tuples.
    """
    # Danbooru category mapping
    CATEGORY_MAP = {
        0: "general",
        1: "artist",
        3: "copyright",
        4: "character",
        5: "meta"
    }
    
    grouped = {cat: [] for cat in CATEGORY_MAP.values()}
    
    for tag in tags:
        if not tag.get("is_deprecated", False):
            category_num = tag.get("category", 0)
            category_name = CATEGORY_MAP.get(category_num, "unknown")
            if category_name in grouped:
                grouped[category_name].append((tag["name"], tag["post_count"]))
    
    # Sort each category by post_count (descending)
    for category in grouped:
        grouped[category].sort(key=lambda x: x[1], reverse=True)
    
    return grouped


def save_category_tags(grouped_tags):
    """
    Saves separate JSON file for each category containing only tag names.
    """
    for category, tag_list in grouped_tags.items():
        # Extract just the tag names (not counts)
        tag_names = [name for name, count in tag_list]
        
        filename = f"danbooru_tags_{category}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(tag_names, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(tag_names)} {category} tags to {filename}")


if __name__ == "__main__":
    print("Starting Danbooru tag scraper...")
    tags = fetch_tags(limit=1000, sleep=1.0)
    print(f"Grouping and filtering {len(tags)} tags by category...")
    grouped_tags = filter_and_group_tags(tags)
    save_category_tags(grouped_tags)
    print("Done!")
