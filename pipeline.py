import re
import os
import json
import time
import requests

import google.generativeai as genai
from pydantic import BaseModel, Field
from typing import List
from youtube_transcript_api import YouTubeTranscriptApi
from geopy.geocoders import Nominatim
from geopy.extra.rate_limiter import RateLimiter

# ---------- Step 1: Transcript Retrieval ----------

def get_video_id_from_url(video_url):
    match = re.search(r"(?:v=|\/)([0-9A-Za-z_-]{11}).*", video_url)
    if not match:
        raise ValueError(f"Invalid YouTube URL: {video_url}")
    return match.group(1)


def get_transcript(video_url):
    video_id = get_video_id_from_url(video_url)
    api = YouTubeTranscriptApi()
    transcript_list = api.list(video_id)

    selected = None
    for t in transcript_list:
        if t.language_code == 'en':
            selected = t
            break

    if not selected:
        available = list(transcript_list)
        if not available:
            raise ValueError("No transcripts available at all for this video.")
        selected = available[0]

    data = selected.fetch()
    return " ".join(item.text for item in data)


# ---------- Step 2: Destination Extraction (AI) ----------

class Destination(BaseModel):
    name: str = Field(description="Clean, geocodable place name, e.g. 'Ella, Sri Lanka'")
    context: str = Field(description="One-line note on why/how it was mentioned in the video")

class DestinationList(BaseModel):
    destinations: List[Destination]


def extract_destinations(transcript_text, gemini_key):
    genai.configure(api_key=gemini_key)
    model = genai.GenerativeModel('models/gemini-flash-lite-latest')

    prompt = f"""
    You are analyzing a travel vlog transcript. Extract every distinct
    real-world travel DESTINATION that is mentioned or visited —
    cities, towns, villages, landmarks, beaches, viewpoints, temples,
    waterfalls, national parks, etc.

    STRICT RULES:
    - Do NOT extract hotel names, restaurant names, or transport methods.
    - Only extract places a user could navigate to and visit.
    - Normalize each name to be geocoding-friendly: add the country or
      region if it helps disambiguate (e.g. "Galle" -> "Galle, Sri Lanka").
    - If the same place is mentioned multiple times, list it only once.
    - The transcript may be in Sinhala or English. Always output names
      and context in English.
    - If nothing qualifies, return an empty list — do not invent places.

    Transcript:
    {transcript_text}
    """

    response = model.generate_content(
        prompt,
        generation_config={
            "response_mime_type": "application/json",
            "response_schema": DestinationList
        }
    )
    return DestinationList.model_validate_json(response.text)


# ---------- Step 3: Geocoding (Free, with Fallback) ----------

geolocator = Nominatim(user_agent="my_travel_app_v1_student_project")
geocode = RateLimiter(geolocator.geocode, min_delay_seconds=1.1)


def geocode_destination_free_with_fallback(place_name):
    attempts = [place_name]
    parts = [p.strip() for p in place_name.split(",")]
    if len(parts) > 1:
        for i in range(1, len(parts)):
            fallback_name = ", ".join(parts[i:])
            if fallback_name not in attempts:
                attempts.append(fallback_name)

    last_error = None
    for attempt_name in attempts:
        try:
            location = geocode(attempt_name)
            if location:
                return {
                    "name": place_name,
                    "success": True,
                    "lat": location.latitude,
                    "lng": location.longitude,
                    "formatted_address": location.address,
                    "resolved_via": attempt_name,
                    "used_fallback": attempt_name != place_name,
                }
        except Exception as e:
            last_error = str(e)
            continue

    return {
        "name": place_name,
        "success": False,
        "error": f"All attempts failed. Last error: {last_error}" if last_error else "No results found even with fallback",
    }


def geocode_all_destinations_free(destination_list):
    results = []
    for d in destination_list.destinations:
        geo = geocode_destination_free_with_fallback(d.name)
        geo["context"] = d.context
        results.append(geo)
        time.sleep(1)
    return results


# ---------- Step 4: Places Enrichment (Overpass API - Free) ----------

OVERPASS_URL = "http://overpass-api.de/api/interpreter"
HEADERS = {"User-Agent": "TravelAssistantApp/1.0 (student project)"}

CATEGORY_TAGS = {
    "restaurant": 'node["amenity"="restaurant"];',
    "hotel": 'node["tourism"="hotel"];',
    "car_rental": '''
        node["amenity"="car_rental"];
        node["shop"="car_rental"];
    ''',
    "attraction": '''
        node["tourism"="attraction"];
        node["tourism"="museum"];
        node["tourism"="viewpoint"];
        node["historic"];
    ''',
}

DEFAULT_RADIUS = {
    "restaurant": 2000,
    "hotel": 2000,
    "attraction": 2000,
    "car_rental": 8000,
}


def find_nearby_places(lat, lon, category, radius=2000, retries=2):
    if category not in CATEGORY_TAGS:
        return {"success": False, "error": f"Unknown category: {category}"}

    tag_block = CATEGORY_TAGS[category]
    tag_lines = [line.strip() for line in tag_block.strip().split(";") if line.strip()]
    queries_with_radius = "\n".join(
        f'{line}(around:{radius},{lat},{lon});' for line in tag_lines
    )

    query = f"""
    [out:json][timeout:25];
    (
    {queries_with_radius}
    );
    out;
    """

    for attempt in range(retries + 1):
        try:
            response = requests.get(OVERPASS_URL, params={"data": query}, headers=HEADERS)

            if response.status_code == 200:
                data = response.json()
                places = []
                seen_names = set()
                for element in data.get("elements", []):
                    tags = element.get("tags", {})
                    name = tags.get("name")
                    if not name or name in seen_names:
                        continue
                    seen_names.add(name)
                    places.append({
                        "name": name,
                        "lat": element.get("lat"),
                        "lon": element.get("lon"),
                    })
                return {"success": True, "places": places}
            else:
                if attempt < retries:
                    time.sleep(5)
                    continue
                return {"success": False, "error": f"Server returned {response.status_code}"}

        except Exception as e:
            if attempt < retries:
                time.sleep(5)
                continue
            return {"success": False, "error": str(e)}


# ---------- Step 5: Navigation Links (Free) ----------

def get_navigation_url(origin_lat, origin_lng, dest_lat, dest_lng, travel_mode="driving"):
    base_url = "https://www.google.com/maps/dir/?api=1"
    return (
        f"{base_url}"
        f"&origin={origin_lat},{origin_lng}"
        f"&destination={dest_lat},{dest_lng}"
        f"&travelmode={travel_mode}"
    )


def add_navigation_to_destinations(geocoded_list, user_lat, user_lng, travel_mode="driving"):
    for dest in geocoded_list:
        if dest.get("success"):
            dest["navigation_url"] = get_navigation_url(
                user_lat, user_lng, dest["lat"], dest["lng"], travel_mode
            )
    return geocoded_list


# ---------- Step 6: Full Pipeline + Caching ----------

CACHE_DIR = "video_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(video_id):
    return os.path.join(CACHE_DIR, f"{video_id}.json")


def load_from_cache(video_id):
    path = get_cache_path(video_id)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def save_to_cache(video_id, data):
    path = get_cache_path(video_id)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def process_video(video_url, gemini_key, user_lat=None, user_lng=None,
                   travel_mode="driving", force_refresh=False):
    video_id = get_video_id_from_url(video_url)

    if not force_refresh:
        cached = load_from_cache(video_id)
        if cached:
            if user_lat and user_lng:
                cached["destinations"] = add_navigation_to_destinations(
                    cached["destinations"], user_lat, user_lng, travel_mode
                )
            return cached

    transcript_text = get_transcript(video_url)
    dest_list = extract_destinations(transcript_text, gemini_key)

    if not dest_list.destinations:
        result = {"video_id": video_id, "video_url": video_url, "destinations": []}
        save_to_cache(video_id, result)
        return result

    geocoded = geocode_all_destinations_free(dest_list)

    for dest in geocoded:
        if not dest.get("success"):
            continue
        dest["nearby"] = {}
        for category, radius in DEFAULT_RADIUS.items():
            place_result = find_nearby_places(dest["lat"], dest["lng"], category, radius=radius)
            dest["nearby"][category] = place_result.get("places", []) if place_result["success"] else []
            time.sleep(0.5)

    result = {
        "video_id": video_id,
        "video_url": video_url,
        "destinations": geocoded,
    }

    save_to_cache(video_id, result)

    if user_lat and user_lng:
        result["destinations"] = add_navigation_to_destinations(
            result["destinations"], user_lat, user_lng, travel_mode
        )

    return result