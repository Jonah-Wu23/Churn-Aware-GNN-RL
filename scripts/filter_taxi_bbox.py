"""Filter NYC taxi trips by bounding box using taxi zone centroids."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import geopandas as gpd
import pandas as pd
import yaml

LOG = logging.getLogger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter NYC taxi trips by bbox")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--zones-path", default="data/external/nyc_taxi_zones/taxi_zones.shp")
    parser.add_argument("--input-glob", default="data/raw/NYC data/*.parquet")
    parser.add_argument("--out-dir", default="data/processed/nyc_bbox")
    parser.add_argument("--north", type=float)
    parser.add_argument("--south", type=float)
    parser.add_argument("--east", type=float)
    parser.add_argument("--west", type=float)
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args(argv)


def load_bbox(args: argparse.Namespace) -> dict[str, float]:
    bbox = None
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        bbox = cfg.get("graph", {}).get("bbox")
    if all(v is not None for v in (args.north, args.south, args.east, args.west)):
        bbox = {
            "north": float(args.north),
            "south": float(args.south),
            "east": float(args.east),
            "west": float(args.west),
        }
    if not isinstance(bbox, dict):
        raise SystemExit("Missing bbox (use --config or --north/--south/--east/--west)")
    for key in ("north", "south", "east", "west"):
        if key not in bbox:
            raise SystemExit(f"Missing bbox.{key}")
    return {
        "north": float(bbox["north"]),
        "south": float(bbox["south"]),
        "east": float(bbox["east"]),
        "west": float(bbox["west"]),
    }


def load_zone_centroids(zones_path: Path) -> pd.DataFrame:
    gdf = gpd.read_file(zones_path)
    if gdf.crs is None:
        raise SystemExit("Taxi zones shapefile missing CRS")
    gdf_proj = gdf.to_crs(epsg=2263)
    gdf_proj["centroid"] = gdf_proj.geometry.centroid
    gdf_centroid = gdf_proj.set_geometry("centroid").to_crs(epsg=4326)
    gdf_centroid["lon"] = gdf_centroid.geometry.x
    gdf_centroid["lat"] = gdf_centroid.geometry.y
    return gdf_centroid[["LocationID", "lon", "lat"]].rename(columns={"LocationID": "location_id"})


def in_bbox(series_lon: pd.Series, series_lat: pd.Series, bbox: dict[str, float]) -> pd.Series:
    return (
        (series_lon >= bbox["west"])
        & (series_lon <= bbox["east"])
        & (series_lat >= bbox["south"])
        & (series_lat <= bbox["north"])
    )


def filter_trip_file(path: Path, out_dir: Path, bbox: dict[str, float], zones: pd.DataFrame) -> None:
    LOG.info("Filtering %s", path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    df = df.rename(columns={"PULocationID": "pu_location_id", "DOLocationID": "do_location_id"})
    if "pu_location_id" not in df.columns or "do_location_id" not in df.columns:
        raise SystemExit(f"Missing PULocationID/DOLocationID in {path}")

    df = df.merge(zones, how="left", left_on="pu_location_id", right_on="location_id")
    df = df.rename(columns={"lon": "pu_lon", "lat": "pu_lat"}).drop(columns=["location_id"])
    df = df.merge(zones, how="left", left_on="do_location_id", right_on="location_id")
    df = df.rename(columns={"lon": "do_lon", "lat": "do_lat"}).drop(columns=["location_id"])

    mask = in_bbox(df["pu_lon"], df["pu_lat"], bbox) & in_bbox(df["do_lon"], df["do_lat"], bbox)
    filtered = df[mask].copy()

    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / path.name
    if out_path.suffix.lower() == ".csv":
        filtered.to_csv(out_path, index=False)
    else:
        filtered.to_parquet(out_path, index=False)

    LOG.info("Wrote %s (%d -> %d rows)", out_path, len(df), len(filtered))


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO))
    bbox = load_bbox(args)
    zones = load_zone_centroids(Path(args.zones_path))

    paths = [Path(p) for p in Path().glob(args.input_glob)]
    if not paths:
        raise SystemExit(f"No files match {args.input_glob}")

    for path in paths:
        filter_trip_file(path, Path(args.out_dir), bbox, zones)


if __name__ == "__main__":
    main()
