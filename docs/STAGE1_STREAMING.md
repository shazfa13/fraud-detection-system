# Stage 1: continuous Kafka–Spark inference

Stage 1 runs a long-lived Spark Structured Streaming query against the existing `fraud-transactions` topic. It loads the API-selected model at startup (`v2` by default; set `FRAUD_ACTIVE_MODEL_VERSION=v1` before launch to select V1), reuses the API's feature builder and risk rules, and never accepts `isFraud` as inference input. Predictions are appended as JSON Lines to `data/stage1_streaming/predictions.jsonl`; rejected events and per-event inference failures go to `data/stage1_streaming/errors.jsonl`. Query heartbeat/status is written to `data/stage1_streaming/stream_status.json`. Checkpoint state is kept at `checkpoints/stage1_kafka_spark`.

The repository's `tools/kafka.tgz` is an archive, not an installed broker; the checked-in `tools/kafka` directory has no Kafka scripts. Use the Kafka installation already configured on the machine. Do not format or reset an existing Kafka data directory. This project's existing broker address is `127.0.0.1:9092` and topic is `fraud-transactions`.

## Start and verify

Run these from the project root in separate PowerShell terminals.

**Terminal 1 — Kafka broker (required):** activate the project environment if needed, then start the existing Kafka KRaft broker using its current configuration and storage:

```powershell
$KafkaHome = 'C:\path\to\existing\kafka'
& "$KafkaHome\bin\windows\kafka-server-start.bat" "$KafkaHome\config\kraft\server.properties"
```

Keep this terminal open. Do not run `kafka-storage.bat format` against existing storage.

**Terminal 2 — topic check/create (one-time per broker):**

```powershell
.\venv\Scripts\Activate.ps1
@'
from kafka.admin import KafkaAdminClient, NewTopic
from src.streaming import config
admin = KafkaAdminClient(bootstrap_servers=config.KAFKA_BOOTSTRAP_SERVERS)
try:
    topics = admin.list_topics()
    if config.KAFKA_TOPIC not in topics:
        admin.create_topics([NewTopic(config.KAFKA_TOPIC, num_partitions=1, replication_factor=1)])
    print(config.KAFKA_TOPIC, "exists; partitions:", admin.describe_topics([config.KAFKA_TOPIC]))
finally:
    admin.close()
'@ | python -
```

This does not delete/reset existing topic data. The stream relies on a single partition to preserve the API's process-local chronological account state.

**Terminal 3 — continuous Spark consumer (required):**

```powershell
.\venv\Scripts\Activate.ps1
$env:FRAUD_ACTIVE_MODEL_VERSION = 'v2'
python -m src.streaming.stage1_streaming
```

Keep this terminal open. Spark uses the configured connector coordinate and may need Maven Central access the first time. Model selection and threshold are loaded by the API config at process startup; changing model selection requires restarting the stream.

**Terminal 4 — synthetic demo producer (optional):**

```powershell
.\venv\Scripts\Activate.ps1
python -m src.streaming.stage1_producer --delay 2
```

Keep this terminal open while publishing. Events are labeled `data_origin=synthetic_paysim_demo`, because PaySim is synthetic. Ctrl+C stops the producer; Ctrl+C in Terminal 3 stops the Spark query cleanly; stop the broker using Ctrl+C in Terminal 1.

Verify output and errors in another shell:

```powershell
Get-Content .\data\stage1_streaming\predictions.jsonl -Tail 10
Get-Content .\data\stage1_streaming\errors.jsonl -Tail 10
Get-Content .\data\stage1_streaming\stream_status.json
```

The dashboard's Streaming Monitor shows recent persisted Stage 1 predictions and the heartbeat state separately from Phase 12 historical evidence. To check checkpoint recovery, stop Spark with Ctrl+C, restart the exact command with the same checkpoint path, and publish another event; the prior offsets are recovered. Do not delete the checkpoint. Kafka/Spark micro-batches and the JSONL append sink provide at-least-once persistence: a process interruption between output append and checkpoint commit can produce a duplicate. Consumers can use `(kafka_partition, kafka_offset)` as the stable delivery key. Account history itself is held in process memory by the existing API feature pipeline and is reinitialized on a full Spark process restart; historical state warm-up is not part of Stage 1.

Errors are persisted and logged. Malformed records are rejected; prediction exceptions are isolated per event. A connector/broker/query-level failure stops the job and records an error in stream status. Existing Phase 7 and Phase 12 output/checkpoint paths remain separate.

Stage 2 adds a secondary SQLite registry after each successful JSONL append. If that registry is unavailable, Stage 1 still retains the prediction in JSONL, logs a `stage2_store_failure`, and continues. After recovery, run `python -m src.labels.cli replay` to import missed JSONL predictions idempotently. See [STAGE2_LABELS.md](STAGE2_LABELS.md).
