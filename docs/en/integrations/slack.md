<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/integrations/slack.md -->
<!-- i18n: source-sha256=11e51c325cd259cbe03ddee2b1da8a43d4008ed90aff26e14165864ae4646efc generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: 581e20f1

# Slack Integration

Slack integration ingests Slack events into Anima's inbox and performs message sending, search, and other actions according to the configuration. Socket Mode is available on servers within a NAT. If using Webhook Mode, an externally reachable HTTPS endpoint is required.

## Preparing the Slack App

Create a Bot in the Slack API App configuration and add the required Bot Token Scopes and receive events. For receive events, specify channel, DM, and group DM messages along with `app_mention` depending on the scope of use. In Socket Mode, enable Socket Mode and issue an App-Level Token with `connections:write`. In Webhook Mode, register `/api/webhooks/slack/events` as the Request URL in Event Subscriptions and prepare a Signing Secret.

Limit the required permissions to the features actually used. Scopes such as read, send, and user information retrieval should be configured according to the events registered in the Slack App and the scope of the tools used.

## Credentials

Use `SLACK_BOT_TOKEN` for the Bot Token, `SLACK_APP_TOKEN` for the Socket Mode App-Level Token, and `SLACK_SIGNING_SECRET` for Webhook signature validation. Register the values in a credential management method such as vault, shared credentials, or environment variables, and do not include actual values in public configuration files or documents. App-Level Token is not required outside of Socket Mode.

## AnimaWorks Configuration

Enable the integration in `external_messaging.slack` of `config.json`, and set `mode` to `socket` or `webhook`. For the shared Bot, `anima_mapping` maps Slack channel IDs to Anima names, and `default_anima` is the default destination when no explicit mapping exists. Channels with empty values can be ignored without being routed to the default Anima. For a list of configuration items, see [external_messaging in the configuration reference](../reference/config.md).

```json
{
  "external_messaging": {
    "slack": {
      "enabled": true,
      "mode": "socket",
      "anima_mapping": {
        "C0123456789": "sample-anima"
      },
      "default_anima": ""
    }
  }
}
```

Check the channel ID from the Slack channel details. Invite the Bot to the channels to receive. If using an individual Bot, confirm that its credentials are registered in the individual Anima's namespace.

## Connection and Verification

After saving the Slack connection configuration, restart the server, or run Slack's hot reload if the configuration supports it. Confirm that the server log shows a successful connection, then send a message from a test channel and verify that it arrives in the correct Anima inbox. For Webhook Mode, verify the public endpoint, signature validation, and event subscription.

If messages do not arrive, check `enabled` and `mode`, credentials, the channel ID to Anima mapping, the Bot's invitation status, and the events and scopes registered in the Slack App in order. For Webhook, check the Signing Secret and the HTTPS path. After confirming receipt, enable external replies and Board synchronization to the minimum necessary extent.
