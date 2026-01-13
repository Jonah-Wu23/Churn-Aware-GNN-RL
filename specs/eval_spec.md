# Evaluation Specification

## Core Metrics
- TACC: Total Avoided Private Car Travel Time
- Algorithmic churn rate (excludes structural_unreachable)
- 95th percentile wait time
- Gini coefficient of service rate across stops

## Structural Unreachability
- Report separately and exclude from algorithmic churn

## Stress Tests
- Surge: demand spike 1.5x
- Bait: center short trips vs edge long trips
- Conflict: onboard passengers vs hotspot demand

## Reporting
- Metrics CSV per run
- JSON summaries under reports/metrics/
- Include config hash and git commit

## Current Implementation Notes
- Episode-level stats are emitted by the Gym env (served, churned, structural_unserviceable, service_gini)
- Unified evaluator and stress test harness are not implemented yet

## TODO
- Specify tolerance bounds for Gym vs SUMO metrics
- Define minimum stress test repetitions per scenario
