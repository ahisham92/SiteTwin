"""Places on Earth, map tiles, and the site's own flat metres.

Everything in a scene is in metres east (x) and north (y) of the model's
origin, which stands at the IFC site's latitude and longitude. Over the
kilometre or so a scene covers, a local tangent plane is exact enough.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EARTH = 6378137.0
TILE = 256


def lonlat_to_merc(lon: float, lat: float) -> tuple[float, float]:
    """Web Mercator metres, the projection every slippy-map tile is cut in."""
    x = EARTH * math.radians(lon)
    y = EARTH * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))
    return x, y


def merc_to_pixel(x: float, y: float, zoom: int) -> tuple[float, float]:
    """Global pixel coordinates at a zoom level (origin top-left of the world)."""
    world = TILE * 2 ** zoom
    half = math.pi * EARTH
    return (x + half) / (2 * half) * world, (half - y) / (2 * half) * world


def lonlat_to_pixel(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    return merc_to_pixel(*lonlat_to_merc(lon, lat), zoom)


def metres_per_pixel(lat: float, zoom: int) -> float:
    return 2 * math.pi * EARTH * math.cos(math.radians(lat)) / (TILE * 2 ** zoom)


@dataclass(frozen=True)
class Local:
    """The site's own frame: metres east and north of (lat, lon)."""
    lat: float
    lon: float

    def to_xy(self, lon: float, lat: float) -> tuple[float, float]:
        x = EARTH * math.cos(math.radians(self.lat)) * math.radians(lon - self.lon)
        y = EARTH * math.radians(lat - self.lat)
        return x, y

    def to_lonlat(self, x: float, y: float) -> tuple[float, float]:
        lon = self.lon + math.degrees(x / (EARTH * math.cos(math.radians(self.lat))))
        lat = self.lat + math.degrees(y / EARTH)
        return lon, lat

    def bbox(self, radius: float) -> tuple[float, float, float, float]:
        """(south, west, north, east) in degrees around the origin."""
        w, s = self.to_lonlat(-radius, -radius)
        e, n = self.to_lonlat(radius, radius)
        return s, w, n, e


def sun_position(lat: float, lon: float, year: int, month: int, day: int, hour_utc: float) -> tuple[float, float]:
    """Solar (elevation, azimuth) in degrees, azimuth clockwise from north (NOAA's approximation)."""
    n = (367 * year - 7 * (year + (month + 9) // 12) // 4 + 275 * month // 9 + day - 730531.5
         + hour_utc / 24.0)
    mean_long = (280.460 + 0.9856474 * n) % 360
    anomaly = math.radians((357.528 + 0.9856003 * n) % 360)
    ecl_long = math.radians(mean_long + 1.915 * math.sin(anomaly) + 0.020 * math.sin(2 * anomaly))
    obliquity = math.radians(23.439 - 0.0000004 * n)
    ra = math.atan2(math.cos(obliquity) * math.sin(ecl_long), math.cos(ecl_long))
    dec = math.asin(math.sin(obliquity) * math.sin(ecl_long))
    gmst = (18.697374558 + 24.06570982441908 * n) % 24
    hour_angle = math.radians((gmst * 15 + lon) % 360) - ra
    phi = math.radians(lat)
    elev = math.asin(math.sin(phi) * math.sin(dec) + math.cos(phi) * math.cos(dec) * math.cos(hour_angle))
    az = math.atan2(-math.sin(hour_angle),
                    math.tan(dec) * math.cos(phi) - math.sin(phi) * math.cos(hour_angle))
    return math.degrees(elev), math.degrees(az) % 360
