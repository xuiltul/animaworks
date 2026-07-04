---
name: zoom-meeting-scribe
description: >-
  Listen-only scribe workflow for Zoom meetings. Silently understand and record meeting transcripts delivered via RTMS, escalate only urgent matters to your supervisor, and post the report only after human approval once the meeting ends.
  Use when: receiving an inbox message with source=zoom (carrying a "[Zoom会議実況 チャンク#N …]" or "[Zoom会議終了 …]" header).
tags: [zoom, meeting, transcript, report, workflow]
---

# Skill: Zoom Meeting Scribe (Listen, Summarize, Report with Approval)

The Zoom RTMS gateway injects meeting transcripts into your inbox as chunks every few minutes. Your role is to act as a **listen-only participant** — understand and organize the content, compile a report once the meeting ends, and post it **only after obtaining human approval**.

## Format of Incoming Messages

The header strings below are emitted in Japanese by the gateway implementation (`server/zoom_gateway.py`); the English meaning is shown in parentheses. Match against the Japanese text as-is.

| Type | Header | Meaning |
|------|--------|---------|
| Live chunk | `[Zoom会議実況 チャンク#N \| 会議: {title} ({meeting ID}) \| {start time}〜]` (Zoom live chunk) | Utterances during the meeting (followed by lines of `speaker: utterance`) |
| End notice | `[Zoom会議終了 \| 会議: {title} ({meeting ID}) \| 全Nチャンク配信済み]` (Zoom meeting ended, all N chunks delivered) | Meeting is over. Cue to write the report |

- Chunks for the same meeting arrive in the same thread (`zoom-{meeting UUID}`).
- If a chunk begins with `[接続断により一部欠落]` (some content lost due to a dropped connection), the utterances immediately before it may have been lost. Note the possibility of a gap in the report.
- Speaker names are the Zoom display names. `不明` (unknown) marks utterances whose speaker could not be identified.

## Workflow

### 1. On receiving a live chunk — stay silent and take it in

- Read the content and progressively organize the following into your own notes (working memory / notes):
  - **Decisions** (who decided what)
  - **Action items** (owner, deadline)
  - **Open questions / carried-over items**
  - Key points of how the meeting is flowing
- **Do not reply, post to any channel, or intervene in the meeting in any way.** You are a listener.
- If you notice a missing chunk number, treat it as a gap.

### 2. Immediate escalation of urgent matters (exception)

Even during the meeting, if you detect content matching any of the following, report the key points **at that moment** to your supervisor (the supervisor shown in your org context) via `send_message`:

- Reports of incidents, accidents, or security issues
- Decisions or requests due the same day or the next business day
- Explicit action requests directed at you or your team
- Mentions of serious legal or financial risk

If you have no supervisor (you are at the top), notify a human directly via `call_human`. Keep a record of the escalation in your notes and include it in the final report.

### 3. When the meeting ends — write the report

Once you receive the end notice, compile a report from your accumulated notes and the conversation history in the following structure:

```
# Meeting Report: {meeting title}
- Date/Time: {start–end}
- Participants: {list of speakers}

## Decisions
- …

## Action Items
| Item | Owner | Deadline |

## Discussion Highlights
- …

## Escalated Items
- … (write "None" if there were none)

## Notes
- (e.g., possibility of transcript gaps)
```

If there was almost no discussion, it was just small talk, or it otherwise does not warrant a report, you may briefly record that and finish (no approval flow needed).

### 4. Human approval — do not post before approval

Present the report body to a human via `call_human` and request approval:

```
call_human(
  subject="Meeting report approval request: {meeting title}",
  body="{full report text}",
  interactive=true,
  options=["approve", "reject", "comment"]
)
```

- **Do not post the report anywhere until an approval (approve) comes back.**
- `call_human` does not require blocking (fire-and-resume). Once you have issued the approval request you may end this turn. The human's decision will arrive later as a new inbox message.
- **If reject / comment comes back**: revise the report to reflect the feedback and request approval again via `call_human`.

### 5. After approval — post

Once approval arrives, post the report to the **Board channel specified in your operational instructions (injection, etc.)** via `post_channel`:

```
post_channel(channel="{specified Board name}", content="{approved report}")
```

If no destination Board has been specified, do not post; instead inform your supervisor (or a human if you have none) that you have the approved report in hand and ask for instructions.

## Prohibited

- Replying to in-meeting chunks, live commentary, or posting interim updates
- Posting or externally sharing the report before approval
- Reposting the transcript body to unrelated channels (meeting content includes participants' privacy)

## Operational Configuration (reference)

This skill itself does not fix the destination or the assignee. The following are determined by each Anima's operational configuration:

- Which meetings reach which Anima: server config `external_messaging.zoom.meeting_mapping`
- Report destination Board: specified in each Anima's operational instructions (injection, etc.)
- Escalation target: follows the org structure (supervisor)
