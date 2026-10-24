# MicroPython / CPython compatible weather fetcher using Open-Meteo.
# On MicroPython, connect to WiFi first (see connect_wifi below).

try:
    import urequests as requests
except ImportError:
    import requests

try:
    import ujson as json
except ImportError:
    import json

LAT = 51.04205429966398
LON = -4.207773277677971
URL = ("https://api.open-meteo.com/v1/forecast?latitude={}&longitude={}"
       "&current_weather=true").format(LAT, LON)

# WMO weather interpretation codes
WEATHER_CODES = {
    0: "Clear sky", 1: "Mainly clear", 2: "Partly cloudy", 3: "Overcast",
    45: "Fog", 48: "Rime fog",
    51: "Light drizzle", 53: "Moderate drizzle", 55: "Dense drizzle",
    56: "Light freezing drizzle", 57: "Dense freezing drizzle",
    61: "Slight rain", 63: "Moderate rain", 65: "Heavy rain",
    66: "Light freezing rain", 67: "Heavy freezing rain",
    71: "Slight snow", 73: "Moderate snow", 75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers", 81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers", 86: "Heavy snow showers",
    95: "Thunderstorm", 96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}

COMPASS = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]


def connect_wifi(ssid, password, timeout=15):
    import network
    import time
    wlan = network.WLAN(network.STA_IF)
    wlan.active(True)
    if not wlan.isconnected():
        wlan.connect(ssid, password)
        start = time.time()
        while not wlan.isconnected():
            if time.time() - start > timeout:
                raise RuntimeError("WiFi connection timed out")
            time.sleep(0.5)
    return wlan.ifconfig()[0]


def compass(degrees):
    return COMPASS[int((degrees % 360) / 22.5 + 0.5) % 16]


def fetch_weather():
    response = requests.get(URL)
    try:
        if response.status_code != 200:
            raise RuntimeError("HTTP error {}".format(response.status_code))
        return json.loads(response.text)
    finally:
        response.close()


def format_weather(data):
    cw = data["current_weather"]
    units = data.get("current_weather_units", {})
    # time is ISO8601, e.g. 2026-10-23T16:45
    date, _, clock = cw["time"].partition("T")
    desc = WEATHER_CODES.get(cw["weathercode"], "Unknown ({})".format(cw["weathercode"]))
    lines = [
        "Weather report",
        "Location:    {:.4f}, {:.4f} (elevation {} m)".format(
            data["latitude"], data["longitude"], data.get("elevation", "?")),
        "Time:        {} {} ({})".format(date, clock, data.get("timezone_abbreviation", "GMT")),
        "Conditions:  {}".format(desc),
        "Temperature: {} {}".format(cw["temperature"], units.get("temperature", "C")),
        "Wind speed:  {} {}".format(cw["windspeed"], units.get("windspeed", "km/h")),
        "Wind from:   {} ({} deg)".format(compass(cw["winddirection"]), cw["winddirection"]),
        "Daylight:    {}".format("Day" if cw.get("is_day") else "Night"),
    ]
    return "\n".join(lines)


def main():
    # On a microcontroller, uncomment and fill in:
    # connect_wifi("YOUR_SSID", "YOUR_PASSWORD")
    try:
        print(format_weather(fetch_weather()))
    except Exception as e:
        print("Error: {}".format(e))


if __name__ == "__main__":
    main()
