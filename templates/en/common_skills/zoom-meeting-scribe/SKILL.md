---
name: zoom-meeting-scribe
description: >-
  Zoom meeting listening workflow. Silently understand and record meeting transcripts received via RTMS, escalate only urgent matters to your supervisor, and post a report after receiving human approval once the meeting ends.
  Use when: When you receive an inbox message with source=zoom (with a "[Zoom meeting live chunk #N …]" or "[Zoom meeting ended …]" header).
tags: [zoom, meeting, transcript, report, workflow]
---


# Skill: Zoom Meeting Scribe (Listening, Summarizing, and Report with Approval)

The Zoom RTMS gateway injects meeting transcripts into your inbox as chunks every few minutes. Your role is to understand and organize the content as a **listener-only participant**, compile a report after the meeting ends, and post it **only after receiving human approval**.

## Format of Incoming Messages

| Type | Header | Meaning |
|------|--------|------|
| Live chunk | `[Zoom会議実況 チャンク#N \| 会議: {題名} ({会議ID}) \| {開始時刻}〜]` | Speech during the meeting (followed by lines of speaker name: statement) |
| End notification | `[Zoom会議終了 \| 会議: {題名} ({会議ID}) \| 全Nチャンク配信済み]` | Meeting ended. Signal to create a report |

- Chunks from the same meeting arrive in the same thread (`zoom-{会議UUID}`)
- If `[接続断により一部欠落]` appears at the start of a chunk, the immediately preceding speech may be lost. Note the possibility of missing content in the report
- Speaker names are Zoom display names. "Unknown" indicates speech where the speaker could not be identified

## Workflow

### 1. On Receiving a Live Chunk — Stay Silent and Take It In

- Read the content and organize it sequentially in your own notes (working memory or notes):
  - **Decisions** (who decided what)
  - **Action items** (owner and deadline)
  - **Open issues and carry-over items**
  - Key points of the meeting flow
- **Do not reply, post to channels, or intervene in the meeting in any way**. You are the listener
- If you notice a missing chunk number, treat it as a gap in the transcript

### 2. Immediate Escalation of Urgent Matters (Exception)

Even during the meeting, if you detect any of the following, report the key points **at that moment** to your supervisor (the supervisor shown in your organization context) via `send_message`:

- Reports of outages, incidents, or security issues
- Decisions or requests with a deadline of today or the next business day
- Explicit action requests addressed to you or your team
- Mentions of significant legal or financial risks

If you have no supervisor (you are at the top), notify a human directly via `call_human`. Record the fact of escalation in your notes and include it in the final report.

### 3. On Meeting End — Create the Report

When you receive the end notification, create a report from your accumulated notes and the conversation history using the following structure:

```
# 会議レポート: {会議名}
- 日時: {開始〜終了}
- 参加者: {発話者一覧}

## 決定事項
- …

## アクションアイテム
| 項目 | 担当 | 期限 |

## 議論の要点
- …

## エスカレーション済み事項
- …（なければ「なし」）

## 備考
- （トランスクリプト欠落の可能性等）
```

If there is almost no speech or only casual chat and the meeting does not warrant a report, you may briefly note that and finish (no approval flow needed).

### 4. Human Approval — Do Not Post Before Approval

Present the report text to a human via `call_human` and request approval:

```
call_human(
  subject="会議レポート承認依頼: {会議名}",
  body="{レポート全文}",
  interactive=true,
  options=["approve", "reject", "comment"]
)
```

- **Do not post the report anywhere until approval (approve) is returned**
- call_human does not require waiting (fire-and-resume). Once you have sent the approval request, you may end this turn. The human's decision will arrive later as a new inbox message
- **If reject / comment is returned**: revise the report to reflect the feedback and request approval again via call_human

### 5. After approval — Post

Once approval arrives, post the report using `animaworks-tool board post <channel> <text>` to the **Board channel specified in the operational instructions (e.g., injection)**:

```
animaworks-tool board post {指定されたBoard名} - <<'EOF'
{承認済みレポート}
EOF
```

If no destination Board is specified, do not post. Tell your supervisor (or a human, if there is no supervisor) that you have the approved report and ask for instructions.

## Prohibited Actions

- Replying to chunks, providing live updates, or posting interim progress during the meeting
- Posting or externally sharing the report before approval
- Reposting transcript content to unrelated channels (meeting content includes participants' private information)

## Operational Configuration (Reference)

This skill itself does not fix the posting destination or responsible person. The following are determined by each Anima's operational configuration:

- Which meetings reach which Anima: server configuration `external_messaging.zoom.meeting_mapping`
- Report posting target Board: specified in each Anima's operational instructions (e.g., injection)
- Escalation destination: follows the organizational structure (supervisor)
