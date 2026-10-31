<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/integrations/zoom.md -->
<!-- i18n: source-sha256=1409948a4743338cc6c2e8ebdb8d5970b6458bf7229ed6f9e1459b7bf3d06632 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Verified commit: 581e20f1

# Zoom RTMS Integration

Zoom RTMS receives meeting transcripts via WebSocket and passes them as chunks to the configured Anima inbox. AnimaWorks does not join meetings or speak through this channel; it processes the received transcriptions.

## Prerequisites

- Configure an RTMS-compatible App and event subscription on the Zoom side.
- Prepare an HTTPS endpoint on the server that is externally reachable for webhooks coming from Zoom.
- Use `ZOOM_SECRET_TOKEN` for signature validation and URL validation, and register `ZOOM_CLIENT_ID` and `ZOOM_CLIENT_SECRET` for RTMS connections as credentials.
- Register AnimaWorks configuration and credentials in a secure location without exposing their values.

Check Zoom contract terms, RTMS availability, and transcription usage conditions in your account and Zoom-side configuration. Also comply with notification and consent requirements for participants.

## AnimaWorks Configuration

Enable `enabled` in `external_messaging.zoom` of `config.json`. `meeting_mapping` is the mapping table from meeting ID to the receiving Anima, and `default_anima` is used for meetings without a match. If neither is assigned, there is no receiving destination. Adjust the chunk interval and character limit according to your use case. See the [configuration reference](../reference/config.md) for items and default values.

```json
{
  "external_messaging": {
    "zoom": {
      "enabled": true,
      "default_anima": "sample-anima",
      "meeting_mapping": {
        "1234567890": "sample-anima"
      },
      "chunk_interval_seconds": 300,
      "chunk_max_chars": 4000
    }
  }
}
```

Save credentials and configuration before starting or restarting the server. Register `/api/webhooks/zoom` as the Zoom webhook URL. Check the signature validation, webhook reception, and RTMS stream connection logs before running a test meeting.

## Operational Checks

Check connection and last reception status using the Zoom gateway information and server logs in `/api/system/health`. In a test meeting, confirm that transcriptions are being generated and that transcript messages arrive in the receiving inbox. Ingestion and Anima response processing are separate stages, so if responses are slow after arriving in the inbox, also check the execution status on the Anima side.

If nothing is received, verify that the webhook is reachable over HTTPS, that URL validation succeeds, that the signature Secret Token and Client ID / Secret match, and that RTMS and transcripts are available on the Zoom side. Also check whether the meeting ID matches `meeting_mapping` and whether `default_anima` is specified.
