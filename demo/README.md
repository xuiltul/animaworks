# AnimaWorks Demo — Try It in 60 Seconds

**[日本語版はこちら](README.ja.md)**

Spin up a fully working AI office with 3 autonomous agents — no setup wizard, no configuration. Just an API key and Docker.

The demo comes pre-loaded with 3 days of activity history, so you'll see a living organization from the moment you open the dashboard.

## Quick Start

### 1. Clone

```bash
git clone https://github.com/xuiltul/animaworks.git
cd animaworks/demo
```

### 2. Set your API key

```bash
cp .env.example .env
# Edit .env and paste your Anthropic API key
```

Get one at [console.anthropic.com](https://console.anthropic.com/) if you don't have one yet.

### 3. Launch

```bash
docker compose up
```

Open **http://localhost:18500** and you're in.

---

## What You'll See

When the dashboard loads, you'll find a 3-person team already at work:

| Agent | Role | What They Do |
|-------|------|-------------|
| **Alex** | Product Manager (leader) | Sets priorities, delegates tasks, reviews progress |
| **Kai** | Lead Engineer | Implements features, investigates technical issues |
| **Nova** | Team Coordinator | Manages schedules, keeps communication flowing |

Alex is in charge. Kai and Nova report to Alex. This hierarchy is fully functional — Alex can delegate tasks to them, check their status, and they report back autonomously.

### Things to try

- **Chat with Alex** — Ask about the team's progress or give a new directive
- **Watch the Activity feed** — See agents communicating in real-time
- **Check the Board** — #general channel has ongoing team discussions
- **Open the 3D Workspace** — See characters sitting at desks and moving around
- **Talk to Kai directly** — Ask him a technical question
- **Wait 5 minutes** — Heartbeats fire and agents start acting on their own

### Pre-loaded history

The demo includes 3 days of simulated activity (auto-adjusted to today's date):

- Activity logs showing past conversations and decisions
- Current tasks in progress
- Messages on the shared #general channel

This means the dashboard won't be empty — you'll see a team with context, history, and ongoing work from the first moment.

---

## Presets

Four presets are available, combining language and personality style:

| Preset | Language | Style | Characters |
|--------|----------|-------|------------|
| `en-anime` (default) | English | Anime-inspired casual | Alex, Kai, Nova |
| `en-business` | English | Professional business | Alex, Kai, Nova |
| `ja-anime` | Japanese | Anime-style casual | Kaito, Sora, Hina |
| `ja-business` | Japanese | Professional business | Kaito, Sora, Hina |

Switch presets with the `PRESET` environment variable:

```bash
PRESET=ja-anime docker compose up
```

> **Note:** The preset is applied on first run only. To switch presets, remove the Docker volume first:
>
> ```bash
> docker compose down -v
> PRESET=ja-business docker compose up
> ```

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | (required) | Your Anthropic API key |
| `PRESET` | `en-anime` | Which preset to use |
| `TZ` | `Asia/Tokyo` | Container timezone |

All variables can be set in the `.env` file or passed directly:

```bash
ANTHROPIC_API_KEY=sk-ant-... PRESET=en-business docker compose up
```

### Ports

The demo server runs on port **18500**. If that's taken:

```bash
# In docker-compose.yml, change the port mapping:
ports:
  - "9000:18500"   # access via http://localhost:9000
```

---

## How the Demo Works

On first launch, the entrypoint script:

1. Initializes the AnimaWorks runtime
2. Creates 3 agents from the selected preset's character sheets
3. Applies preset-specific configuration (heartbeat interval, etc.)
4. Copies pre-built character assets (avatars)
5. Loads 3 days of example activity data with auto-adjusted timestamps
6. Starts the server

Subsequent launches skip initialization and use the existing data in the Docker volume.

### Autonomous Behavior

Once running, the agents operate autonomously:

- **Heartbeat** — Every 5 minutes (demo interval), each agent reviews their situation and decides what to do
- **Cron tasks** — Scheduled tasks defined per agent (daily summaries, monitoring, etc.)
- **Delegation chains** — Alex delegates to Kai/Nova, they execute and report back
- **Board activity** — Agents post updates to shared channels

You don't need to do anything — just watch. Or jump in and give them new instructions.

---

## Data Persistence

Agent data is stored in a Docker volume (`animaworks-demo-data`). Your conversations, agent memories, and activity logs survive container restarts.

To start completely fresh:

```bash
docker compose down -v    # removes the volume
docker compose up         # fresh initialization
```

---

## Troubleshooting

### "Animas will not be able to respond"

Your `ANTHROPIC_API_KEY` is missing or invalid. Check your `.env` file:

```bash
cat .env   # should show ANTHROPIC_API_KEY=sk-ant-api03-...
```

### Port 18500 already in use

Another service is using that port. Either stop it or change the mapping:

```bash
# Check what's using the port
lsof -i :18500

# Or change the port in docker-compose.yml
ports:
  - "9000:18500"
```

### Agents aren't responding

- Verify your API key is valid (test it at [console.anthropic.com](https://console.anthropic.com/))
- Check container logs: `docker compose logs -f`
- Ensure you have API credits available

### Container build fails

```bash
# Rebuild from scratch
docker compose build --no-cache
docker compose up
```

### Want to reset everything

```bash
docker compose down -v
docker compose up
```

---

## Next Steps

Ready to build your own AI organization?

- **Full install** — See the [main README](../README.md) for native installation
- **Create your own agents** — Write a character sheet in Markdown and the framework does the rest
- **Add more LLMs** — AnimaWorks supports Claude, GPT, Gemini, local models, and more
- **Explore the docs** — [Design Philosophy](../docs/vision.md) · [Memory System](../docs/memory.md) · [Security](../docs/security.md)

---

*This demo is part of [AnimaWorks](https://github.com/xuiltul/animaworks) — an open-source framework for building autonomous AI organizations.*
