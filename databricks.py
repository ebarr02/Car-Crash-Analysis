"""Load Ticketmaster events for Rogers Arena and BC Place."""

import json
import math
import os
import re
import ssl
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import quote, unquote, urlencode, urlparse
from urllib.request import Request, urlopen
from html import unescape

import pandas as pd

try:
    import certifi
except ImportError:  # pragma: no cover
    certifi = None


BASE_URL = "https://app.ticketmaster.com/discovery/v2"
API_KEY_ENV_VAR = "TICKETMASTER_API_KEY"
SYSTEM_CA_FILE = Path("/etc/ssl/cert.pem")
EVENTS_CSV_PATH = Path("ticketmaster_events.csv")
WEATHER_CSV_PATH = Path("vancouver_event_weather.csv")
WIKIMEDIA_PAGEVIEWS_URL = "https://wikimedia.org/api/rest_v1/metrics/pageviews"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
BC_PLACE_EVENTS_API = "https://www.bcplace.com/wp-json/wp/v2/event"
ROGERS_ARENA_SETLIST_URL = (
    "https://www.setlist.fm/venue/rogers-arena-vancouver-bc-canada-33d65801.html"
)


def _api_key() -> str:
    api_key = os.getenv(API_KEY_ENV_VAR)
    if not api_key:
        raise RuntimeError(
            f"Set the {API_KEY_ENV_VAR} environment variable before loading events."
        )
    return api_key


def _ssl_context() -> ssl.SSLContext:
    cafile = SYSTEM_CA_FILE if SYSTEM_CA_FILE.is_file() else None
    if cafile is None and certifi is not None:
        cafile = certifi.where()
    return ssl.create_default_context(cafile=cafile)


def _load_json(url: str, headers: dict[str, str] | None = None, timeout: int = 120) -> dict[str, Any]:
    request = Request(url, headers=headers or {})
    with urlopen(request, timeout=timeout, context=_ssl_context()) as response:
        return json.load(response)


def _load_text(url: str, timeout: int = 120) -> str:
    request = Request(url, headers={"User-Agent": "ActuarialEventAnalysis/1.0"})
    with urlopen(request, timeout=timeout, context=_ssl_context()) as response:
        return response.read().decode("utf-8")


def _get_json(path: str, params: dict[str, Any]) -> dict[str, Any]:
    query = urlencode({"apikey": _api_key(), **params})
    return _load_json(f"{BASE_URL}{path}?{query}")


def get_venue_id(name: str) -> str:
    """Return the Ticketmaster venue ID for an exact venue-name search."""
    data = _get_json("/venues.json", {"keyword": name, "size": 1})
    venues = data.get("_embedded", {}).get("venues", [])
    if not venues:
        raise LookupError(f"No Ticketmaster venue found for {name!r}.")
    return venues[0]["id"]


def get_events(
    venue_id: str,
    start: str | None = None,
    end: str | None = None,
    size: int = 200,
) -> list[dict[str, Any]]:
    """Return every Ticketmaster event for a venue within the date range."""
    if not 1 <= size <= 200:
        raise ValueError("size must be between 1 and 200.")

    params: dict[str, Any] = {"venueId": venue_id, "size": size}
    if start:
        params["startDateTime"] = start
    if end:
        params["endDateTime"] = end

    events: list[dict[str, Any]] = []
    page = 0
    while True:
        data = _get_json("/events.json", {**params, "page": page})
        events.extend(data.get("_embedded", {}).get("events", []))

        page_info = data.get("page", {})
        if page >= page_info.get("totalPages", 0) - 1:
            break
        page += 1

    return events


def load_venue_events(
    start: str = "2025-01-01T00:00:00Z",
) -> dict[str, list[dict[str, Any]]]:
    """Load events for the two Vancouver venues used by this project."""
    venues = ("Rogers Arena", "BC Place")
    return {
        venue: get_events(get_venue_id(venue), start=start)
        for venue in venues
    }


def _historical_event(date_value: str, name: str, location: str, time: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "dates": {"start": {"localDate": date_value, "localTime": time}},
        "_historical": True,
        "location": location,
    }


def load_bc_place_history() -> list[dict[str, Any]]:
    """Load dated historical event posts retained by BC Place's public API."""
    events = []
    for page in range(1, 21):
        posts = _load_json(f"{BC_PLACE_EVENTS_API}?per_page=100&page={page}")
        if not posts:
            break
        for post in posts:
            text = unescape(re.sub(r"<[^>]+>", " ", post["content"]["rendered"]))
            match = re.search(r"([A-Z][a-z]+\s+\d{1,2}(?:st|nd|rd|th)?\s*,\s*20(?:25|26|27))", text)
            if not match:
                continue
            date_text = re.sub(r"(?<=\d)(st|nd|rd|th)", "", match.group(1))
            try:
                event_date = datetime.strptime(date_text, "%B %d, %Y")
            except ValueError:
                event_date = datetime.strptime(date_text, "%b %d, %Y")
            date_value = event_date.date().isoformat()
            time_match = re.search(r"\b(\d{1,2}:\d{2}\s*(?:AM|PM))\b", text)
            time = ""
            if time_match:
                time_text = re.sub(r"\s*([AP]M)$", r" \1", time_match.group(1))
                time = datetime.strptime(time_text, "%I:%M %p").strftime("%H:%M:%S")
            events.append(_historical_event(date_value, unescape(post["title"]["rendered"]), "BC Place", time))
        if len(posts) < 100:
            break
    return events


def load_rogers_arena_concert_history() -> list[dict[str, Any]]:
    """Load publicly recorded 2025+ Rogers Arena concerts from Setlist.fm."""
    events = []
    pattern = re.compile(
        r'href="\.\./show/(20(?:25|26|27)-\d{2}-\d{2})/rogers-arena-vancouver-bc-canada-[^"]+"[^>]* title="View this ([^"]+) show"'
    )
    for page in range(1, 51):
        html = _load_text(f"{ROGERS_ARENA_SETLIST_URL}?page={page}")
        matches = pattern.findall(html)
        if not matches:
            break
        events.extend(_historical_event(date_value, unescape(name), "Rogers Arena") for date_value, name in matches)
        if all(date_value < "2025-01-01" for date_value, _ in matches):
            break
    return events


def load_historical_venue_events() -> dict[str, list[dict[str, Any]]]:
    return {
        "Rogers Arena": load_rogers_arena_concert_history(),
        "BC Place": load_bc_place_history(),
    }


def _wikipedia_article(event: dict[str, Any]) -> str | None:
    attractions = event.get("_embedded", {}).get("attractions", [])
    if not attractions:
        return None

    wiki_links = attractions[0].get("externalLinks", {}).get("wiki", [])
    if not wiki_links:
        return None

    path = urlparse(wiki_links[0].get("url", "")).path
    if not path.startswith("/wiki/"):
        return None
    return unquote(path.removeprefix("/wiki/")).replace(" ", "_")


def _wikipedia_pageviews(article: str) -> int | None:
    end_date = date.today() - timedelta(days=1)
    start_date = end_date - timedelta(days=29)
    pageviews_url = (
        f"{WIKIMEDIA_PAGEVIEWS_URL}/per-article/en.wikipedia.org/all-access/"
        f"all-agents/{quote(article, safe='')}/daily/"
        f"{start_date:%Y%m%d}/{end_date:%Y%m%d}"
    )
    try:
        data = _load_json(
            pageviews_url,
            headers={"User-Agent": "ActuarialEventAnalysis/1.0"},
        )
    except (HTTPError, URLError, json.JSONDecodeError):
        return None
    return sum(item["views"] for item in data.get("items", []))


def popularity_scores(events: list[dict[str, Any]]) -> dict[str, int]:
    """Return relative 1-10 popularity scores from 30-day Wikipedia page views."""
    article_by_event_id = {
        event["id"]: _wikipedia_article(event)
        for event in events
        if event.get("id")
    }
    articles = sorted({article for article in article_by_event_id.values() if article})
    with ThreadPoolExecutor(max_workers=8) as executor:
        pageviews_by_article = dict(
            zip(articles, executor.map(_wikipedia_pageviews, articles), strict=True)
        )

    event_pageviews = {
        event_id: pageviews_by_article.get(article)
        for event_id, article in article_by_event_id.items()
    }
    available_pageviews = [
        views for views in event_pageviews.values() if views is not None
    ]
    if not available_pageviews:
        return {event_id: 1 for event_id in event_pageviews}

    minimum = math.log1p(min(available_pageviews))
    maximum = math.log1p(max(available_pageviews))

    def score(views: int | None) -> int:
        if views is None:
            return 1
        if minimum == maximum:
            return 5
        return round(1 + 9 * (math.log1p(views) - minimum) / (maximum - minimum))

    return {event_id: score(views) for event_id, views in event_pageviews.items()}


def _duration_hours(event: dict[str, Any]) -> float | str:
    dates = event.get("dates", {})
    start = dates.get("start", {})
    end = dates.get("end", {})
    if not all(
        (
            start.get("localDate"),
            start.get("localTime"),
            end.get("localDate"),
            end.get("localTime"),
        )
    ):
        return ""

    start_time = datetime.fromisoformat(f"{start['localDate']}T{start['localTime']}")
    end_time = datetime.fromisoformat(f"{end['localDate']}T{end['localTime']}")
    duration = (end_time - start_time).total_seconds() / 3600
    return round(duration, 2) if duration >= 0 else ""


def save_events_to_csv(
    venue_events: dict[str, list[dict[str, Any]]],
    output_path: Path = EVENTS_CSV_PATH,
) -> None:
    """Save the event details needed for attendance analysis to a CSV file."""
    events = [
        event
        for venue_events_list in venue_events.values()
        for event in venue_events_list
    ]
    scores = popularity_scores(events)
    records = [
        {
            "date": event.get("dates", {}).get("start", {}).get("localDate", ""),
            "time": event.get("dates", {}).get("start", {}).get("localTime", ""),
            "event_name": event.get("name", ""),
            "location": venue,
            "expected_popularity_score": scores.get(event.get("id"), 1),
            "expected_attendance": event.get("expectedAttendance", ""),
            "actual_attendance": event.get("actualAttendance", ""),
            "duration_hours": _duration_hours(event),
        }
        for venue, venue_events_list in venue_events.items()
        for event in venue_events_list
    ]
    columns = [
        "date",
        "time",
        "event_name",
        "location",
        "expected_popularity_score",
        "expected_attendance",
        "actual_attendance",
        "duration_hours",
    ]
    data = pd.DataFrame(records, columns=columns).replace("", pd.NA)
    data.dropna(axis=1, how="all").sort_values(
        ["date", "time"], ascending=False, na_position="last"
    ).to_csv(output_path, index=False)


def merge_venue_events(
    primary: dict[str, list[dict[str, Any]]],
    historical: dict[str, list[dict[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    merged = {}
    for venue in primary:
        seen = set()
        merged[venue] = []
        for event in primary[venue] + historical.get(venue, []):
            start = event.get("dates", {}).get("start", {})
            key = (start.get("localDate"), event.get("name", "").casefold())
            if key not in seen:
                seen.add(key)
                merged[venue].append(event)
    return merged


WEATHER_CONDITIONS = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog", 51: "Light drizzle", 53: "Moderate drizzle",
    55: "Heavy drizzle", 61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow", 80: "Rain showers",
    81: "Moderate rain showers", 82: "Violent rain showers", 95: "Thunderstorm",
}


def create_weather_dataframe(
    events_path: Path | None = None,
    output_path: Path = WEATHER_CSV_PATH,
    start_date: date | str = date(2025, 8, 1),
) -> pd.DataFrame:
    """Build a Vancouver-only hourly weather dataset from August 2025 onward."""
    if isinstance(start_date, str):
        start_date = date.fromisoformat(start_date)

    start = start_date
    end = date.today() + timedelta(days=15)
    all_hourly: list[pd.DataFrame] = []

    historical_end = date.today() - timedelta(days=1)
    chunk_days = 31
    if start <= historical_end:
        cursor = start
        while cursor <= min(end, historical_end):
            chunk_end = min(cursor + timedelta(days=chunk_days - 1), min(end, historical_end))
            params = {
                "latitude": 49.2827,
                "longitude": -123.1207,
                "start_date": cursor.isoformat(),
                "end_date": chunk_end.isoformat(),
                "hourly": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                "timezone": "America/Vancouver",
            }
            historical = _load_json(f"{OPEN_METEO_ARCHIVE_URL}?{urlencode(params)}")["hourly"]
            historical_df = pd.DataFrame(historical)
            historical_df["datetime"] = pd.to_datetime(historical_df["time"])
            historical_df["date"] = historical_df["datetime"].dt.strftime("%Y-%m-%d")
            historical_df["time"] = historical_df["datetime"].dt.strftime("%H:%M:%S")
            all_hourly.append(historical_df)
            cursor = chunk_end + timedelta(days=1)

    if end >= date.today():
        forecast_start = max(start, date.today())
        forecast_end = min(end, date.today() + timedelta(days=15))
        if forecast_start <= forecast_end:
            cursor = forecast_start
            while cursor <= forecast_end:
                chunk_end = min(cursor + timedelta(days=15), forecast_end)
                params = {
                    "latitude": 49.2827,
                    "longitude": -123.1207,
                    "start_date": cursor.isoformat(),
                    "end_date": chunk_end.isoformat(),
                    "hourly": "temperature_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m",
                    "timezone": "America/Vancouver",
                }
                forecast = _load_json(f"{OPEN_METEO_FORECAST_URL}?{urlencode(params)}")["hourly"]
                forecast_df = pd.DataFrame(forecast)
                forecast_df["datetime"] = pd.to_datetime(forecast_df["time"])
                forecast_df["date"] = forecast_df["datetime"].dt.strftime("%Y-%m-%d")
                forecast_df["time"] = forecast_df["datetime"].dt.strftime("%H:%M:%S")
                all_hourly.append(forecast_df)
                cursor = chunk_end + timedelta(days=1)

    if not all_hourly:
        raise ValueError("No Vancouver weather data could be retrieved for the selected date range.")

    weather = (
        pd.concat(all_hourly, ignore_index=True)
        .drop_duplicates(subset=["datetime"])
        .sort_values("datetime")
        .reset_index(drop=True)
    )
    weather["city"] = "Vancouver"
    weather["weather_condition"] = weather["weather_code"].map(WEATHER_CONDITIONS)
    weather["temperature_2m"] = pd.to_numeric(weather["temperature_2m"], errors="coerce")
    weather["apparent_temperature"] = pd.to_numeric(weather["apparent_temperature"], errors="coerce")
    weather["precipitation"] = pd.to_numeric(weather["precipitation"], errors="coerce")
    weather["wind_speed_10m"] = pd.to_numeric(weather["wind_speed_10m"], errors="coerce")

    columns = [
        "datetime",
        "date",
        "time",
        "city",
        "weather_condition",
        "temperature_2m",
        "apparent_temperature",
        "precipitation",
        "wind_speed_10m",
        "weather_code",
    ]
    weather[columns].to_csv(output_path, index=False)
    return weather[columns]


if __name__ == "__main__":
    if os.getenv(API_KEY_ENV_VAR):
        venue_events = merge_venue_events(load_venue_events(), load_historical_venue_events())
        save_events_to_csv(venue_events)
        for venue, events in venue_events.items():
            print(f"{venue}: {len(events)} events loaded")
        print(f"Saved event data to {EVENTS_CSV_PATH.resolve()}")
    elif EVENTS_CSV_PATH.exists():
        print(f"{API_KEY_ENV_VAR} not set; using existing event CSV at {EVENTS_CSV_PATH.resolve()}.")
    else:
        raise RuntimeError(
            f"Set the {API_KEY_ENV_VAR} environment variable or provide {EVENTS_CSV_PATH} before running this script."
        )

    weather_df = create_weather_dataframe()
    print(f"Saved weather data to {WEATHER_CSV_PATH.resolve()}")
    print(weather_df.head().to_string(index=False))
