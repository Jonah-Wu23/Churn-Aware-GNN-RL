# Data Specification

## Sources
- OSM extracts for the target city
- Taxi OD datasets (e.g., NYC Open Data)
- NYC TLC taxi zones shapefile (for LocationID -> centroid lookup)

## Raw OD Schema (example)
- tpep_pickup_datetime (timestamp)
- pu_lon, pu_lat (float)
- do_lon, do_lat (float)
- (optional) structural_unreachable (bool)

## Mapped OD Output
- tpep_pickup_datetime
- pickup_stop_id, dropoff_stop_id
- pickup_walk_time_sec, dropoff_walk_time_sec
- structural_unreachable (bool)

## Mapping Policy
- Use KNN by Euclidean distance to select candidate stops
- Use pedestrian network shortest path to choose the nearest feasible stop
- Mark structural_unreachable when walk time > 10 minutes
- OD filtering (NYC bbox study area): keep trips where pickup and dropoff zone centroids fall within the bbox
- Barrier impact: straight-line distance < 50 m but pedestrian-walk distance > 500 m
- Unreachability threshold: walk time > 600 sec using walk_speed_mps = 1.4
- soft_assignment_delta_sec is configurable for near-ties in walking time

## Audit Outputs
- reports/audit/od_mapping.json
- Include mismatch_rate, barrier_impact_count, and structural_unreachability stats

## Versioning
- `data/manifest.md` records source URLs, hashes, and extraction dates
- All processed datasets include a schema version field

## TODO
- Confirm required time zone and time granularity
- Define OD request filtering rules for outliers
