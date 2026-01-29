"""Filter NYC taxi trips by bounding box using taxi zone centroids."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import geopandas as gpd
import pandas as pd
from pyproj import Transformer
from shapely.geometry import box
import yaml

LOG = logging.getLogger(__name__)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter NYC taxi trips by bbox")
    parser.add_argument("--config", help="Path to YAML config file")
    parser.add_argument("--zones-path", default="data/external/nyc_taxi_zones/taxi_zones.shp")
    parser.add_argument("--input-glob", default="data/raw/NYC data/*.parquet")
    parser.add_argument("--out-dir", default="data/processed/nyc_bbox")
    parser.add_argument("--seed", type=int, default=7)
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


def load_zone_geometries(zones_path: Path) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(zones_path)
    if gdf.crs is None:
        raise SystemExit("Taxi zones shapefile missing CRS")
    gdf = gdf.rename(columns={"LocationID": "location_id"})
    gdf["location_id"] = pd.to_numeric(gdf["location_id"], errors="coerce").astype("Int64")
    gdf = gdf.dropna(subset=["location_id"]).copy()
    gdf["location_id"] = gdf["location_id"].astype(int)
    return gdf.to_crs(epsg=2263)[["location_id", "geometry"]].copy()


def _bbox_polygon_2263(bbox: dict[str, float]) -> object:
    poly4326 = box(float(bbox["west"]), float(bbox["south"]), float(bbox["east"]), float(bbox["north"]))
    series = gpd.GeoSeries([poly4326], crs="EPSG:4326").to_crs(epsg=2263)
    return series.iloc[0]


def _sample_points_in_geom(geom: object, n: int, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    import shapely

    minx, miny, maxx, maxy = geom.bounds
    xs = np.empty((0,), dtype=float)
    ys = np.empty((0,), dtype=float)
    remaining = int(n)
    attempts = 0
    while remaining > 0:
        attempts += 1
        batch = int(max(1024, remaining * 4))
        cand_x = rng.uniform(minx, maxx, size=batch)
        cand_y = rng.uniform(miny, maxy, size=batch)
        pts = shapely.points(cand_x, cand_y)
        inside = shapely.contains(geom, pts)
        if np.any(inside):
            xs = np.concatenate([xs, cand_x[inside]])
            ys = np.concatenate([ys, cand_y[inside]])
            remaining = int(n - len(xs))
        if attempts > 200:
            break
    if len(xs) < n:
        raise SystemExit("Failed to sample points inside zone geometry")
    return xs[:n], ys[:n]


def _assign_random_points(
    df: pd.DataFrame,
    id_col: str,
    zones_by_id: dict[int, object],
    rng: np.random.Generator,
    transformer: Transformer,
) -> Tuple[np.ndarray, np.ndarray]:
    lon = np.full(len(df), np.nan, dtype=float)
    lat = np.full(len(df), np.nan, dtype=float)
    for loc_id, idxs in df.groupby(id_col).groups.items():
        geom = zones_by_id.get(int(loc_id))
        if geom is None:
            continue
        xs, ys = _sample_points_in_geom(geom, len(idxs), rng)
        out_lon, out_lat = transformer.transform(xs, ys)
        lon[np.array(list(idxs), dtype=int)] = out_lon
        lat[np.array(list(idxs), dtype=int)] = out_lat
    return lon, lat


def in_bbox(series_lon: pd.Series, series_lat: pd.Series, bbox: dict[str, float]) -> pd.Series:
    return (
        (series_lon >= bbox["west"])
        & (series_lon <= bbox["east"])
        & (series_lat >= bbox["south"])
        & (series_lat <= bbox["north"])
    )


def filter_trip_file(
    path: Path,
    out_dir: Path,
    bbox: dict[str, float],
    zones: gpd.GeoDataFrame,
    bbox_poly_2263: object,
    rng: np.random.Generator,
) -> None:
    LOG.info("Filtering %s", path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_parquet(path)

    df = df.rename(columns={"PULocationID": "pu_location_id", "DOLocationID": "do_location_id"})
    if "pu_location_id" not in df.columns or "do_location_id" not in df.columns:
        raise SystemExit(f"Missing PULocationID/DOLocationID in {path}")

    df["pu_location_id"] = pd.to_numeric(df["pu_location_id"], errors="coerce").astype("Int64")
    df["do_location_id"] = pd.to_numeric(df["do_location_id"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["pu_location_id", "do_location_id"]).copy()
    df["pu_location_id"] = df["pu_location_id"].astype(int)
    df["do_location_id"] = df["do_location_id"].astype(int)

    zones_clip = zones.copy()
    zones_clip["geometry"] = zones_clip.geometry.intersection(bbox_poly_2263)
    zones_clip = zones_clip[~zones_clip.geometry.is_empty].copy()
    eligible = set(zones_clip["location_id"].astype(int).tolist())
    mask_zone = df["pu_location_id"].isin(eligible) & df["do_location_id"].isin(eligible)
    filtered = df[mask_zone].copy().reset_index(drop=True)

    zones_by_id = {int(row.location_id): row.geometry for row in zones_clip.itertuples(index=False)}
    transformer = Transformer.from_crs(2263, 4326, always_xy=True)

    pu_lon, pu_lat = _assign_random_points(filtered, "pu_location_id", zones_by_id, rng, transformer)
    do_lon, do_lat = _assign_random_points(filtered, "do_location_id", zones_by_id, rng, transformer)
    filtered["pu_lon"] = pu_lon
    filtered["pu_lat"] = pu_lat
    filtered["do_lon"] = do_lon
    filtered["do_lat"] = do_lat
    filtered = filtered.dropna(subset=["pu_lon", "pu_lat", "do_lon", "do_lat"]).copy()

    mask = in_bbox(filtered["pu_lon"], filtered["pu_lat"], bbox) & in_bbox(
        filtered["do_lon"], filtered["do_lat"], bbox
    )
    filtered = filtered[mask].copy().reset_index(drop=True)

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
    rng = np.random.default_rng(int(args.seed))
    zones = load_zone_geometries(Path(args.zones_path))
    bbox_poly_2263 = _bbox_polygon_2263(bbox)

    paths = [Path(p) for p in Path().glob(args.input_glob)]
    if not paths:
        raise SystemExit(f"No files match {args.input_glob}")

    for path in paths:
        filter_trip_file(path, Path(args.out_dir), bbox, zones, bbox_poly_2263, rng)


if __name__ == "__main__":
    main()
