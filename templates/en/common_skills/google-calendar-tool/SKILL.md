---
name: google-calendar-tool
description: >-
  Google Calendar integration tool. List upcoming events and create new events.
  OAuth2 authentication for direct Calendar API access.
  "calendar" "schedule" "google calendar" "event" "appointment"
tags: [calendar, google, schedule, external]
---

# Google Calendar Tool

External tool for Google Calendar event management via OAuth2 API access.

## Invocation via Bash

Use **Bash** with `animaworks-tool google_calendar <subcommand> [args]`. See Actions below for syntax.

## Actions

### list — List upcoming events
```json
{"tool_name": "google_calendar", "action": "list", "args": {"max_results": 20, "days": 7, "calendar_id": "primary"}}
```

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| max_results | integer | 20 | Maximum events to return |
| days | integer | 7 | Number of days ahead to search |
| calendar_id | string | "primary" | Calendar ID |

### add — Create new event
```json
{"tool_name": "google_calendar", "action": "add", "args": {"summary": "Meeting", "start": "2026-03-04T10:00:00Z", "end": "2026-03-04T11:00:00Z", "description": "Weekly standup", "location": "Room A"}}
```

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| summary | string | Yes | Event title |
| start | string | Yes | Start time (ISO8601 or YYYY-MM-DD for all-day) |
| end | string | Yes | End time (ISO8601 or YYYY-MM-DD for all-day) |
| description | string | No | Event description |
| location | string | No | Event location |
| calendar_id | string | No | Calendar ID (default: primary) |
| attendees | array | No | List of attendee email addresses |

## CLI Usage (S/C/D/G-mode)

```bash
animaworks-tool google_calendar list [-n 20] [-d 7] [--calendar-id primary]
animaworks-tool google_calendar add "Meeting" --start 2026-03-04T10:00:00Z --end 2026-03-04T11:00:00Z
```

## Notes

- OAuth2 authentication flow required on first use
- Place credentials.json at ~/.animaworks/credentials/google_calendar/
- For all-day events, use YYYY-MM-DD format for start/end
