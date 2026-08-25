# Gated tool usage scan

- data_dir: `~/.animaworks`
- animas: 13
- elapsed_s: 4.3

## Per-Anima permissions (external_tools)

| anima | allow_all | allow | deny |
|---|---|---|---|
| aoi | True | — | machine |
| ayame | True | — | machine |
| kotoha | True | slack_send, slack_channel_post | machine |
| mei | True | slack_send, slack_channel_post | machine |
| mio | True | — | machine |
| nagi | True | — | machine |
| natsume | True | — | machine |
| rin | True | — | machine |
| ritsu | True | slack_send | machine |
| sakura | True | slack_send, slack_channel_post | machine |
| sora | True | — | machine |
| sumire | True | slack_send, slack_channel_post | machine |
| yoru | True | — | machine |

## activity_log needle counts (signals, not exact call tallies)

| anima | chatwork_send | discord_send | github_create-issue | github_create-pr | machine_run |
|---|---|---|---|---|---|
| aoi | 91 | 2 | 88 | 88 | 65 |
| ayame | 26 | 0 | 1 | 1 | 182 |
| kotoha | 4131 | 0 | 12 | 12 | 300 |
| mei | 2319 | 4 | 39 | 39 | 1050 |
| mio | 32 | 12 | 6 | 7 | 173 |
| nagi | 6 | 3 | 0 | 0 | 84 |
| natsume | 20 | 12 | 375 | 379 | 4318 |
| rin | 58 | 9 | 1899 | 1864 | 4129 |
| ritsu | 86 | 3 | 1 | 1 | 373 |
| sakura | 949 | 492 | 567 | 552 | 1159 |
| sora | 8 | 6 | 50 | 49 | 1983 |
| sumire | 7 | 0 | 186 | 188 | 22076 |
| yoru | 2 | 0 | 0 | 0 | 1 |

## Recommended allow additions (usage > 0 and not already allowed/denied)

Note: `machine_run` is intentionally never recommended (live animas deny the `machine` tool).

- **chatwork_send**: aoi, ayame, kotoha, mei, mio, nagi, natsume, rin, ritsu, sakura, sora, sumire, yoru
- **discord_send**: aoi, mei, mio, nagi, natsume, rin, ritsu, sakura, sora
- **github_create-issue**: aoi, ayame, kotoha, mei, mio, natsume, rin, ritsu, sakura, sora, sumire
- **github_create-pr**: aoi, ayame, kotoha, mei, mio, natsume, rin, ritsu, sakura, sora, sumire
- **machine_run**: (none)

## PdM allow policy (pi-fix2, 2026-08-01)

- `chatwork_send`: animas with chatwork send usage (exclude sumire/ayame/yoru when usage is noise-only / PdM list)
- `machine_run`: nobody
- `discord_send` / `github_create-*`: usage-based only (see recommended list above)
