import os

"""
Copy/merge these fields into your existing config.py (or compare with yours).
Keep your API keys here (do NOT commit secrets).
"""

# OpenWeather
OWM_API_KEY = os.getenv("OWM_API_KEY", "")
CITY_NAME = "Madrid"
COUNTRY_CODE = "ES"
DAYS_BACK = 10
UNITS = "metric"

# Reddit (source2)
SUBREDDIT = "datascience"
REDDIT_LIMIT = 50
REDDIT_LISTING = "top"
REDDIT_TIME = "year"
REDDIT_USER_AGENT = "BIU-DS21-assignment/1.0 (contact: student)"
