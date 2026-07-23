# TripoSplat Work Dashboard

PostgreSQL-backed dashboard for TripoSplat CPU optimization progress, experiments,
artifacts, documents, and compute-resource snapshots. It listens on port `10101`.

## Data isolation

- PostgreSQL database: `triposplat_dashboard`
- PostgreSQL role: `triposplat_app`
- The database is independent from the `boatrace` database.
- Runtime credentials stay in `dashboard/config/` and are excluded from Git.

## Runtime layout

```text
/workspace/3dgs/triposplat-dashboard/
  .venv/                 dedicated Python virtual environment
  config/dashboard.env   DSN and pgpass path
  config/triposplat.pgpass
  data/seed.json         idempotent initial data
  logs/                  dashboard logs
  run/dashboard.pid      process state
```

## Management

```bash
triposplat-dashboard-admin init
triposplat-dashboard-admin seed data/seed.json
triposplat-dashboard-admin snapshot --workspace /workspace
triposplat-dashboard-admin health

bash scripts/start-dashboard.sh
bash scripts/stop-dashboard.sh
```

The UI is read-only. Work updates are applied through the management CLI so that
progress changes remain explicit and auditable.
