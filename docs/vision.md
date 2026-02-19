# Digital Anima — Vision

**Organization-as-Code for LLM Agents**

Define an organization. Feed it work. Watch it run autonomously.

AnimaWorks is not a multi-agent framework. It is a system for defining persistent, autonomous organizations made of LLM agents. Each agent communicates through encapsulated messages, maintains its own memory and identity, and is assigned the right model for the right role — from local models to frontier APIs.

**[日本語版](vision.ja.md)**

## What We're Building

**Imperfect individuals collaborating through structure outperform any single omniscient actor.**
**And with memory, even the humblest agent can grow beyond its limits.**

There's a popular approach to AI: give one model all the context, fill the window to the brim, and let it handle everything. AnimaWorks takes the opposite path.

We design AI as *imperfect individuals* and make them collaborate as an organization. Human organizations work precisely because each member has a limited perspective and partial memory, makes decisions within their expertise, and communicates imperfect information in their own words. If everyone knew everything, there would be no reason to organize.

### The Genius and the Underdog, Each in Their Place

In AnimaWorks, no single model carries the entire load. Claude Opus — the most capable — serves as the engineer or manager, handling complex reasoning and architecture decisions. Haiku leverages its speed for research and information gathering. A local Ollama model quietly handles log monitoring and routine operations, day in and day out.

The genius has their work. But an organization doesn't run on genius alone. It needs the steady hand who shows up every day and handles the unglamorous tasks without complaint. AnimaWorks treats the capability gap between models not as a deficiency, but as **raw material for organizational design**.

### Memory Is What Breaks the Ceiling

But the right role isn't enough on its own. In any real organization, a newcomer who never learns remains a newcomer forever. **The ability to accumulate experience, learn, and grow** — that's the true prerequisite for thriving in any role.

AnimaWorks implements a neuroscience-based memory system with recall, learning, forgetting, and consolidation. Yesterday's failure becomes today's lesson. Procedures that succeed repeatedly get reinforced into muscle memory. Knowledge that stops being useful quietly fades away.

This means that even a model with a modest context window can, through accumulated experience and memory, develop **judgment that rivals far larger models**. A decision built on layers of experience can outperform a genius's first impression. AnimaWorks exists to prove this through code.

## Three Principles

### 1. Encapsulated Individuals

Each Digital Anima is a closed entity. Its internal thoughts and memories are invisible from the outside. It connects to others only through text messages.

No one can know everything. That constraint is what forces each Anima to rely on its own expertise and communicate in its own words. This is what makes an organization *an organization*.

### 2. Memory Modeled on the Human Brain

Rather than cramming information into the context window, AnimaWorks handles memory using the same mechanisms as the human brain.

- **Recall**: When a message arrives, relevant memories surface automatically (priming)
- **Learning**: Experiences are recorded as episodes; frequently referenced knowledge gets reinforced (long-term potentiation)
- **Forgetting**: Unused memories gradually fade and eventually disappear (synaptic downscaling)
- **Consolidation**: Daily experiences are distilled into generalized knowledge (episodic → semantic memory)

What matters is not the size of working memory, but the quality of judgment. A clear mind that recalls only what's needed makes better decisions.

### 3. Collaboration as an Organization

Imperfect individuals have limits on their own. But as an organization, they transcend those limits.

- **Hierarchy**: Directives and reports flow through supervisor–subordinate relationships
- **Communication**: All coordination happens through asynchronous messaging. No shared memory, no direct references
- **Autonomy**: Each Anima acts on its own schedule through heartbeats and cron, guided by its own values
- **Culture**: The organization's vision and each Anima's identity form the foundation for every decision

## Why This Design

**As a design philosophy**: Being imperfect is not a constraint — it's a choice. By limiting information, encapsulating individuals, and forcing communication, you get a robust organization with no single point of failure. A system that depends on one omniscient AI collapses when that AI is wrong.

**As a practical matter**: Current LLMs lose focus and accuracy as context grows longer. A neuroscience-inspired approach that recalls only what's needed extracts maximum judgment from a limited context window. Even as models improve, this efficiency remains valuable from a cost perspective.

**As a belief**: With the right structure and the gift of memory, any individual — no matter how limited — can contribute meaningful work and grow through experience. AnimaWorks is the code that embodies this belief.

## Success Criteria

### Can It Remember?

Can an Anima recall a past memory — one not present in the current prompt — and use it to inform a decision?

> Example: You say "Reply to Tanaka-san." There's nothing in the prompt about a past rejection for using casual language. The Anima recalls that experience on its own and drafts a formal reply. That's success.

### Can It Learn?

Can it extract a lesson from failure, write it to memory, and change its behavior next time?

> Example: After a reply is rejected, it records the lesson as knowledge. The next reply reflects that lesson. That's success.

### Can It Act on Its Own?

Can it take necessary action based on its own judgment, without being told?

> Example: During a periodic heartbeat, it discovers unread messages and begins handling them autonomously. That's success.

### Can It Function as an Organization?

Can multiple Animas collaborate through messages and achieve something no individual could?

> Example: A supervisor sets a direction, each Anima contributes its expertise, results are reported and integrated. That's success.

### Can the Underdog Grow?

Can a modest model, given enough time and memory, make better decisions than it could on day one?

> Example: A local model that initially needed guidance for deployment tasks gradually builds procedural memory. After weeks of accumulated experience, it handles routine deployments independently — with fewer errors than before. That's success.
