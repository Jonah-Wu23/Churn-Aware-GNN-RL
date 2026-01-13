import argparse
import csv
import os
import time
from urllib.parse import quote

import requests


BASE_URL = "https://data.cityofchicago.org/resource/ajtu-isnz.csv"
SELECT_FIELDS = """\
SELECT
  `trip_id`,
  `taxi_id`,
  `trip_start_timestamp`,
  `trip_end_timestamp`,
  `trip_seconds`,
  `trip_miles`,
  `pickup_census_tract`,
  `dropoff_census_tract`,
  `pickup_community_area`,
  `dropoff_community_area`,
  `fare`,
  `tips`,
  `tolls`,
  `extras`,
  `trip_total`,
  `payment_type`,
  `company`,
  `pickup_centroid_latitude`,
  `pickup_centroid_longitude`,
  `pickup_centroid_location`,
  `dropoff_centroid_latitude`,
  `dropoff_centroid_longitude`,
  `dropoff_centroid_location`
"""


def build_query(start_ts: str, end_ts: str, limit: int | None, offset: int | None) -> str:
    query = f"""{SELECT_FIELDS}
WHERE
  `trip_start_timestamp`
    BETWEEN "{start_ts}" :: floating_timestamp
    AND "{end_ts}" :: floating_timestamp
ORDER BY `trip_start_timestamp` DESC NULL FIRST
""".strip()
    if limit is not None:
        query += f" LIMIT {limit}"
    if offset is not None:
        query += f" OFFSET {offset}"
    return query


def request_csv(url: str, headers: dict[str, str], timeout: int, retries: int) -> list[str]:
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp.text.splitlines()
        except Exception as exc:
            if attempt == retries:
                raise
            wait = min(30, attempt * 2)
            print(f"Retry {attempt}/{retries} after error: {exc}. Waiting {wait}s")
            time.sleep(wait)
    return []


def count_rows(start_ts: str, end_ts: str, headers: dict[str, str], retries: int) -> int:
    count_query = f"""\
SELECT count(*) AS count
WHERE
  `trip_start_timestamp`
    BETWEEN "{start_ts}" :: floating_timestamp
    AND "{end_ts}" :: floating_timestamp
""".strip()
    url = f"{BASE_URL}?$query={quote(count_query)}"
    lines = request_csv(url, headers, timeout=60, retries=retries)
    if len(lines) < 2:
        raise RuntimeError(f"Unexpected count response: {lines}")
    return int(lines[1].strip().strip('"'))


def existing_row_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path, "r", newline="", encoding="utf-8") as f_in:
        for idx, _ in enumerate(f_in):
            pass
    if "idx" not in locals():
        return 0
    return max(0, idx - 1)


def download(start_ts: str, end_ts: str, output_path: str, page_size: int, retries: int) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    headers = {"Accept": "text/csv"}

    total_rows = count_rows(start_ts, end_ts, headers, retries)
    print(f"Total rows: {total_rows}")
    if total_rows == 0:
        with open(output_path, "w", newline="", encoding="utf-8") as f_out:
            f_out.write("")
        return

    existing_rows = existing_row_count(output_path)
    print(f"Existing rows: {existing_rows}")

    mode = "a" if existing_rows > 0 else "w"
    with open(output_path, mode, newline="", encoding="utf-8") as f_out:
        writer = csv.writer(f_out) if mode == "a" else None
        offset = existing_rows
        while offset < total_rows:
            query = build_query(start_ts, end_ts, page_size, offset)
            url = f"{BASE_URL}?$query={quote(query)}"
            lines = request_csv(url, headers, timeout=180, retries=retries)
            if not lines:
                break
            reader = csv.reader(lines)
            header = next(reader, None)
            if header is None:
                break
            if writer is None:
                writer = csv.writer(f_out)
                writer.writerow(header)
            for row in reader:
                writer.writerow(row)
            offset += page_size
            print(f"Downloaded {min(offset, total_rows)} / {total_rows}")
            time.sleep(0.2)

    print(f"Saved to {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download Chicago taxi trips (2024-) with a date range.",
    )
    parser.add_argument(
        "--start",
        required=True,
        help="Start timestamp (e.g. 2025-05-01T00:00:00)",
    )
    parser.add_argument(
        "--end",
        required=True,
        help="End timestamp (e.g. 2025-05-31T23:45:00)",
    )
    parser.add_argument(
        "--output",
        default=os.path.join("data", "external", "Chicago_data.csv"),
        help="Output CSV path (default: data/external/Chicago_data.csv)",
    )
    parser.add_argument(
        "--page-size",
        type=int,
        default=5000,
        help="Rows per request page (default: 5000)",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=8,
        help="Retry attempts for network errors (default: 8)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    download(args.start, args.end, args.output, args.page_size, args.retries)


if __name__ == "__main__":
    main()
