# Engram — Cron Configurations

Copy-paste ready OpenClaw cron JSON for all recommended schedules.

## Prerequisites

Set these environment variables in your OpenClaw config or shell profile:

```bash
export BRAIN_DIR=~/.engram
export ENGRAM_LLM_ENDPOINT=http://localhost:3456/v1/chat/completions
export PYTHONPATH=/path/to/engram-openclaw-skill:$PYTHONPATH
```

---

## Hourly: Extract from memory notes

All memory extraction instances can run concurrently. Each is checkpointed independently.

```json
{
  "name": "engram-extract-memory",
  "schedule": "0 * * * *",
  "command": "cd /path/to/engram-openclaw-skill && python3 -m extractors.memory",
  "timeout": 300,
  "enabled": true
}
```

## Every 4 hours: Extract from session logs

```json
{
  "name": "engram-extract-sessions",
  "schedule": "0 */4 * * *",
  "command": "cd /path/to/engram-openclaw-skill && python3 -m extractors.sessions --limit 20",
  "timeout": 600,
  "enabled": true
}
```

## Every 4 hours: Extract from chat exports

```json
{
  "name": "engram-extract-chats",
  "schedule": "30 */4 * * *",
  "command": "cd /path/to/engram-openclaw-skill && python3 -m extractors.chats",
  "timeout": 600,
  "enabled": true
}
```

## Every 6 hours: Extract from work documents

```json
{
  "name": "engram-extract-work",
  "schedule": "0 */6 * * *",
  "command": "cd /path/to/engram-openclaw-skill && python3 -m extractors.work",
  "timeout": 600,
  "enabled": true
}
```

## 2x daily: Ruminate + Sleep cycle (5am + 5pm local time)

The core consolidation loop. Ruminate generates insights, sleep scores fitness and GCs.

```json
{
  "name": "engram-ruminate-morning",
  "schedule": "0 5 * * *",
  "command": "cd /path/to/engram-openclaw-skill && bash scripts/run_cycles.sh 5",
  "timeout": 1800,
  "enabled": true
}
```

```json
{
  "name": "engram-ruminate-evening",
  "schedule": "0 17 * * *",
  "command": "cd /path/to/engram-openclaw-skill && bash scripts/run_cycles.sh 5",
  "timeout": 1800,
  "enabled": true
}
```

## Every 6 hours: WAL-safe database backup

```json
{
  "name": "engram-backup",
  "schedule": "0 */6 * * *",
  "command": "mkdir -p ${BRAIN_DIR:-~/.engram}/backups && sqlite3 ${BRAIN_DIR:-~/.engram}/brain.db \".backup ${BRAIN_DIR:-~/.engram}/backups/brain-$(date +%Y%m%d-%H%M%S).db\" && find ${BRAIN_DIR:-~/.engram}/backups -name 'brain-*.db' -mtime +7 -delete",
  "timeout": 120,
  "enabled": true
}
```

## Weekly: Full audit + compact

Full decay scan + aggressive GC. Runs Sunday at 3am.

```json
{
  "name": "engram-weekly-compact",
  "schedule": "0 3 * * 0",
  "command": "cd /path/to/engram-openclaw-skill && python3 -m brain.sleep --full --threshold 0.03",
  "timeout": 3600,
  "enabled": true
}
```

## Weekly: Dashboard export

Regenerate the dashboard data. Runs Sunday at 4am.

```json
{
  "name": "engram-dashboard-export",
  "schedule": "0 4 * * 0",
  "command": "cd /path/to/engram-openclaw-skill && python3 -m dashboard.export",
  "timeout": 120,
  "enabled": true
}
```

---

## All crons at a glance

| Schedule | Name | What |
|----------|------|------|
| `0 * * * *` | extract-memory | Daily notes → nodes |
| `0 */4 * * *` | extract-sessions | Session logs → nodes |
| `30 */4 * * *` | extract-chats | Chat exports → nodes |
| `0 */6 * * *` | extract-work | Work docs → nodes |
| `0 5,17 * * *` | ruminate | Think→sleep cycles (2x daily) |
| `0 */6 * * *` | backup | WAL-safe DB backup |
| `0 3 * * 0` | weekly-compact | Full audit + aggressive GC |
| `0 4 * * 0` | dashboard-export | Regenerate visualization data |
