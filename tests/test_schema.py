from pathlib import Path

import pandas as pd


def test_od_mapped_schema():
    base = Path("data/processed/od_mapped")
    files = sorted(base.glob("*.parquet"))
    assert files, "No od_mapped parquet files found"

    nodes = pd.read_parquet("data/processed/graph/layer2_nodes.parquet", columns=["gnn_node_id"])
    valid_stop_ids = set(nodes["gnn_node_id"].astype(int).tolist())

    required_cols = {
        "tpep_pickup_datetime",
        "pickup_stop_id",
        "dropoff_stop_id",
        "pickup_walk_time_sec",
        "dropoff_walk_time_sec",
        "structural_unreachable",
    }

    for path in files:
        df = pd.read_parquet(path, columns=list(required_cols)).head(2000)
        missing = required_cols - set(df.columns)
        assert not missing, f"{path} missing columns: {missing}"

        assert df["pickup_walk_time_sec"].ge(0).all()
        assert df["dropoff_walk_time_sec"].ge(0).all()
        assert df["pickup_stop_id"].isin(valid_stop_ids).all()
        assert df["dropoff_stop_id"].isin(valid_stop_ids).all()
