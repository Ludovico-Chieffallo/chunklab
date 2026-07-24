# Vertex Analytics REST API Reference

Base URL: `https://api.vertex-analytics.example/v2`. All requests and responses use JSON encoded as UTF-8. Timestamps are RFC 3339 in UTC. Identifiers are opaque strings prefixed by resource type (`ds_`, `job_`, `mdl_`, `prd_`, `whk_`). This reference covers API version 2; version 1 remains available at `/v1` until its sunset date announced in the changelog.

## Authentication

### Creating and using API keys

Every request must carry an API key in the `Authorization` header using the `Bearer` scheme. Keys are created in the console under Settings → API keys, are shown once at creation, and are stored by the platform only as salted hashes. A key is bound to one project and one role at creation; the role cannot be changed afterwards, only a new key can be issued.

```bash
curl https://api.vertex-analytics.example/v2/datasets \
  -H "Authorization: Bearer va_live_9f2c8e71d4a05b36" \
  -H "Content-Type: application/json"
```

Keys carry one of three roles. `reader` may perform only GET requests. `writer` may additionally create and update resources but may not delete them. `admin` may delete resources and manage webhooks. Requests made with an insufficient role fail with HTTP 403 and error code `insufficient_role`, and the response names the minimum role required in the `required_role` field.

**Key rotation.**

Two keys can be active on the same project simultaneously to allow zero-downtime rotation: create the replacement key, deploy it everywhere, then revoke the old key. Revocation propagates to all API edges within sixty seconds. The `last_used_at` field on each key, visible in the console and through `GET /keys`, shows the most recent authenticated request and is the recommended check before revoking.

### Session tokens for browser use

API keys must never be embedded in browser or mobile code. For user-facing applications, exchange a server-held key for a short-lived session token with `POST /auth/session-tokens`, passing the allowed scopes and an optional `ttl_seconds` between 60 and 3600, default 900. The response contains a `token` that may be used from the browser with the same `Bearer` scheme and expires automatically; session tokens can be revoked collectively per project but not individually.

```json
{
  "scopes": ["predictions:create", "datasets:read"],
  "ttl_seconds": 600
}
```

## Rate Limiting and Errors

### Rate limits

Each project has a request budget per minute that depends on its plan: 300 requests per minute on Starter, 1,200 on Growth, and 6,000 on Scale. Prediction endpoints have a separate budget counted in prediction units rather than requests. Every response carries three headers: `X-RateLimit-Limit` with the per-minute budget, `X-RateLimit-Remaining` with the requests left in the current window, and `X-RateLimit-Reset` with the epoch second at which the window resets.

When the budget is exhausted the API returns HTTP 429 with error code `rate_limited` and a `Retry-After` header stating the seconds to wait. Clients should honor `Retry-After` and apply jittered exponential backoff; retrying earlier only extends the throttled window. Burst allowances above the nominal budget are granted for up to ten seconds when the previous five minutes stayed under fifty percent utilization.

### Error envelope

Every error response uses one envelope. The `type` field is a stable machine-readable code, `message` is human-readable and may change, `request_id` identifies the request in support conversations, and `param` names the offending field for validation errors.

```json
{
  "error": {
    "type": "validation_failed",
    "message": "split_ratio must be between 0.05 and 0.5",
    "param": "split_ratio",
    "request_id": "req_7d1f0a44"
  }
}
```

Stable error types are: `validation_failed` (422), `not_found` (404), `insufficient_role` (403), `rate_limited` (429), `conflict` (409) for concurrent modification detected through the `If-Match` header, `payload_too_large` (413), and `internal` (500). Integrations must branch on `type`, never on `message`. Every 5xx response is safe to retry with backoff because all mutating endpoints accept an `Idempotency-Key` header whose effect is stored for twenty-four hours.

## Datasets

### Upload and registration

`POST /datasets` registers a dataset. Small files up to 100 MB may be sent inline as multipart form data with the `file` part; larger data must be staged through `POST /datasets/uploads`, which returns a presigned upload URL valid for one hour, followed by registration referencing the `upload_id`. Supported formats are CSV, Parquet, and JSON Lines; compressed variants with gzip or zstd are detected by extension.

```bash
curl -X POST https://api.vertex-analytics.example/v2/datasets \
  -H "Authorization: Bearer $KEY" \
  -F "file=@churn_q3.parquet" \
  -F 'metadata={"name":"churn-q3","schema_mode":"infer"}'
```

With `schema_mode` set to `infer`, column types are inferred from a sample of up to 100,000 rows; set it to `strict` and provide a `schema` object to reject rows that do not match instead. Registration is asynchronous: the response returns the dataset in state `validating`, and the state moves to `ready` or `invalid`. Validation failures attach a `validation_report` listing per-column error counts and up to fifty sample offending rows.

**Dataset states.**

A dataset is in exactly one state: `validating`, `ready`, `invalid`, or `archived`. Only `ready` datasets can be referenced by training jobs and prediction requests. Archiving with `POST /datasets/{id}/archive` detaches no history but blocks new references; archived data is retained for ninety days and then deleted permanently, and `unarchive` is possible only inside that window.

### Querying and versions

`GET /datasets` lists datasets with cursor pagination: pass `limit` up to 100, follow `next_cursor` until it is null. Every mutation of a dataset's data creates a new immutable version, numbered from 1; `GET /datasets/{id}/versions` lists them, and any reference may pin `@N` to a version, as in `ds_8a1b@4`. An unpinned reference always resolves to the latest `ready` version at the time the referencing job starts, and the resolved version is recorded on the job for reproducibility.

## Jobs

### Training jobs

`POST /jobs` starts a training job. The request names the `dataset` (pinned or unpinned), the `target` column, the `task` (`classification` or `regression`), and an optional `preset` controlling the search space: `fast` explores a small space in minutes, `balanced` is the default, `thorough` runs the widest search and may take hours on large data.

```json
{
  "dataset": "ds_8a1b@4",
  "target": "churned",
  "task": "classification",
  "preset": "balanced",
  "split": {"strategy": "stratified", "test_ratio": 0.2},
  "budget_minutes": 60
}
```

The `budget_minutes` field caps total optimization time between 5 and 480; the job stops at the budget with the best model found so far, and `stopped_reason` records whether the budget or convergence ended the search. Class imbalance is handled automatically for classification when the minority class falls below ten percent, and the chosen technique is reported in the training report.

**Job lifecycle and control.**

A job moves through `queued`, `running`, and one terminal state among `succeeded`, `failed`, and `canceled`. `POST /jobs/{id}/cancel` requests cooperative cancellation: a running job checkpoints and stops within two minutes, and a queued job cancels immediately. Jobs emit progress events consumable through webhooks or `GET /jobs/{id}/events`, which returns a bounded stream of the most recent 1,000 events with cursor pagination. A failed job carries `failure_code` and, when the cause is data-related, references the offending dataset version.

### Evaluation reports

Every succeeded training job attaches an evaluation report at `GET /jobs/{id}/report`. For classification it contains accuracy, precision, recall, F1 per class, the ROC AUC, and the confusion matrix at the default threshold; for regression it contains RMSE, MAE, and R². The report also includes feature importances computed by permutation on the held-out split, capped at the top fifty features, and a calibration section stating the Brier score and whether Platt scaling was applied.

## Models

### Registry and stages

A succeeded training job produces a model in the registry. Models carry a `stage` among `candidate`, `production`, and `retired`; exactly one model per project may be in `production` for a given endpoint alias. `POST /models/{id}/promote` moves a candidate to production atomically, demoting the previous production model to `retired` in the same operation, and the promotion is recorded with actor and timestamp in the audit log.

**Model metadata.**

`GET /models/{id}` returns the training dataset version, the preset, the training duration, the metrics of the evaluation report, the schema of expected input features with types and allowed ranges, and a `signature` hash that changes whenever the input schema changes. Clients should pin integrations to the `signature` and treat a change as a breaking event requiring re-validation of the calling pipeline.

### Exporting models

`POST /models/{id}/export` produces a downloadable artifact in ONNX format for supported model families, returning a presigned URL valid for twenty-four hours. Exports are watermarked with the model id and the exporting project, and the export event is written to the audit log. Models trained with the `thorough` preset on ensembles above one gigabyte are not exportable and return `validation_failed` with `param` set to `model_family`.

## Predictions

### Online predictions

`POST /predictions` scores records synchronously against the production model of the named alias. The body carries `alias` and `records`, an array of up to 500 feature objects; the response preserves order and returns, per record, the prediction, the probability per class for classification, and the model id that served it. The p99 latency target for batches of up to 100 records is 150 milliseconds within one region.

```json
{
  "alias": "churn-scorer",
  "records": [
    {"tenure_months": 27, "plan": "growth", "tickets_90d": 3},
    {"tenure_months": 4, "plan": "starter", "tickets_90d": 11}
  ]
}
```

Records failing schema validation do not fail the whole batch: each offending record yields an inline error object with `type` `validation_failed` and the `param` naming the feature, while valid records are scored normally. This partial-failure contract is the reason the endpoint returns HTTP 200 even when some records fail; check per-record `error` fields rather than the status code.

**Explanation of individual predictions.**

Passing `"explain": true` adds, per record, the top contributing features with signed attribution values computed by a sampling variant of SHAP. Explanations roughly triple the latency and double the prediction-unit cost of the request, and they are unavailable for aliases serving models exported and re-imported from outside the platform.

### Batch predictions

`POST /predictions/batches` scores a whole dataset asynchronously: the request names the `alias` and the input `dataset`, and the completed batch writes an output dataset with the original columns plus `prediction`, per-class probabilities, and optionally explanations. Batches respect a separate queue with a per-project concurrency of two, and their prediction-unit price is forty percent below the online rate. A batch over a dataset in a non-`ready` state fails immediately with `conflict`.

## Webhooks

### Subscriptions and delivery

`POST /webhooks` subscribes an HTTPS endpoint to event types: `job.succeeded`, `job.failed`, `dataset.ready`, `dataset.invalid`, `model.promoted`, and `batch.completed`. Each delivery is a POST with the event in the body and two headers: `X-Vertex-Event-Id`, unique per event, and `X-Vertex-Signature`, an HMAC-SHA256 of the raw body computed with the subscription's signing secret returned once at creation.

Consumers must verify the signature against the raw request body before parsing, compare it in constant time, and de-duplicate by event id, because deliveries are at-least-once. Delivery is retried with exponential backoff for up to twenty-four hours; after the final failure the event is parked and visible under `GET /webhooks/{id}/dead-letter` for seven days. Endpoints answering with a status other than 2xx within ten seconds count as failed; heavy processing belongs in a queue behind the receiver, not in the request handler.

**Rotating signing secrets.**

`POST /webhooks/{id}/rotate-secret` returns a new signing secret while keeping the previous one valid for a grace window of twenty-four hours, during which deliveries are signed with both secrets in two headers, `X-Vertex-Signature` for the new and `X-Vertex-Signature-Previous` for the old. Verify against either during the window, then drop the old secret.

## Feature Store

### Feature groups

`POST /feature-groups` defines a named group of features computed from a source dataset or data connection, keyed by one or more entity columns and an event-time column. The definition carries the transformation SQL, the refresh `schedule` in cron syntax evaluated in UTC, and an `online` flag controlling whether the group is materialized to the low-latency store in addition to the offline store. Feature groups are versioned like datasets: editing the transformation creates a new version, and downstream references may pin one.

```json
{
  "name": "customer_activity",
  "entities": ["customer_id"],
  "event_time": "activity_at",
  "sql": "SELECT customer_id, activity_at, logins_7d, tickets_30d FROM {{ source }}",
  "schedule": "0 */6 * * *",
  "online": true
}
```

Materialization runs are visible as jobs of type `materialization` and respect the same lifecycle and events as training jobs. A failed materialization leaves the previous successful snapshot serving, so consumers never observe a half-written refresh; the `freshness_at` field on the group states the event time up to which data is guaranteed complete.

**Point-in-time correctness.**

Training-set assembly with `POST /feature-groups/training-frames` performs a point-in-time join: for each labeled row, the features returned are those whose event time is the latest at or before the label's timestamp, which prevents leakage from the future. The request names the label dataset, the timestamp column, and the feature groups to join; the response is a new dataset in state `validating` like any registration. Rows whose entities have no feature history at the label time receive nulls and are counted in the assembly report rather than dropped silently.

### Online feature retrieval

`POST /feature-groups/{id}/read-online` returns the latest materialized feature vector for up to 200 entity keys per call, with a p99 latency target of 25 milliseconds in-region. Keys absent from the online store return `found: false` rather than an error, and the response carries the group version and the materialization timestamp so callers can enforce their own staleness budget. Online reads are billed as prediction units at one tenth of the online prediction rate.

## Data Connections

### Creating connections

`POST /connections` registers an external source for dataset imports and feature-group sources. Supported connectors are PostgreSQL, MySQL, Snowflake, BigQuery, and S3-compatible object stores. Credentials are written once, stored in the platform secrets manager, and never returned by any read endpoint; `GET /connections/{id}` exposes only the redacted host, database, and the credential's fingerprint. Connections are validated at creation with a permission probe, and the probe's findings are attached as `capabilities`, for example whether the credential may read information schema or only named tables.

```json
{
  "type": "postgresql",
  "host": "warehouse.internal.example",
  "port": 5432,
  "database": "analytics",
  "username": "vertex_reader",
  "password": "•••",
  "ssl_mode": "verify-full"
}
```

**Sync schedules and incremental imports.**

A connection-backed dataset declares a `sync` block with a cron `schedule` and a `mode`: `full` re-imports the whole query result, while `incremental` requires a monotonically increasing `cursor_column` and imports only rows beyond the last recorded cursor. Incremental syncs that detect a cursor regression, for example after a source table rebuild, fail with `conflict` and require an explicit `POST /datasets/{id}/resync-full` to re-baseline. Every sync appends a new dataset version, and the sync history with row counts and durations is available under `GET /datasets/{id}/syncs`.

### Connection health

`GET /connections/{id}/health` runs a live probe measuring reachability, authentication, and read latency, and returns the last ten scheduled-sync outcomes. A connection failing three consecutive scheduled syncs is marked `degraded`, its dependent feature-group refreshes are paused, and a `connection.degraded` webhook event is emitted; recovery is automatic on the next successful probe or sync.

## Monitoring and Drift

### Enabling monitors

`POST /monitors` attaches monitoring to a serving alias. A monitor computes, over a sliding window, the population stability index per feature against the training distribution, the share of null and out-of-range values per feature, and, when ground-truth labels are later uploaded, the realized accuracy metrics. The `window` may be `1d`, `7d`, or `28d`, and thresholds are configured per metric with a warning and a critical level.

```json
{
  "alias": "churn-scorer",
  "window": "7d",
  "thresholds": {
    "psi": {"warning": 0.1, "critical": 0.25},
    "null_rate": {"warning": 0.02, "critical": 0.1}
  }
}
```

Threshold crossings emit `monitor.warning` and `monitor.critical` webhook events, at most one per metric per window to avoid storms, and appear on the alias page with the offending features ranked by severity. Monitors add no latency to predictions because they consume a sampled asynchronous copy of the traffic; the sampling rate adapts between one and one hundred percent to keep the window's sample above ten thousand records where traffic allows.

**Uploading ground truth.**

`POST /monitors/{id}/labels` uploads realized outcomes as pairs of prediction id and label, individually or as a dataset reference for bulk backfill. Labels arriving within ninety days of the prediction are joined to it; older labels are rejected with `validation_failed`. Once at least one thousand labeled predictions exist in a window, the monitor computes realized precision, recall, and calibration drift, and these series appear alongside the input-drift series with the same alerting semantics.

## Projects and Members

### Invitations and roles

`POST /projects/{id}/invitations` invites a member by email with a role among `viewer`, `editor`, and `owner`; the invitation expires after fourteen days and can be resent at most once per day. Role changes and removals are immediate, and the removed member's personal API keys on the project are revoked within five minutes, mirroring console behavior. `GET /projects/{id}/members` lists members with their role, join date, and last activity, and supports the same cursor pagination as every list endpoint.

**Transferring a project.**

`POST /projects/{id}/transfer` moves a project to another organization: the initiator must be an owner on both sides, the transfer is prepared asynchronously, and both organizations' audit logs record the operation with a shared transfer id. Billing attribution switches at the top of the hour following completion, and API keys, webhooks, and connections move unchanged, while member access is re-derived from the destination organization's policies, which may reduce it.

## Audit and Usage

### Audit log

`GET /audit-events` returns the project's control-plane history: key creations and revocations, dataset registrations and archivals, job submissions and cancellations, model promotions, webhook changes, and export events. Each entry carries the actor (member or API key), source IP, action, resource, and a monotonically increasing sequence number safe for incremental consumption. Audit entries are immutable and readable for 400 days, matching the platform-wide audit retention.

### Usage metering

`GET /usage` aggregates consumption by day and by metric: API requests, prediction units, training minutes, and stored gigabyte-hours, with a `group_by` of `project` or `api_key`. Figures for the current day are provisional and settle within six hours after midnight UTC. The endpoint accepts `from` and `to` dates spanning at most ninety-two days per request, and its output reconciles exactly with the monthly invoice lines of the same period.
