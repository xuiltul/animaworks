# AnimaWorks Brain Mapping — Architecture Mapped to the Human Brain

**[日本語版](brain-mapping.ja.md)**

> Created: 2026-02-19
> Related: [vision.md](vision.md), [memory.md](memory.md)

> **Note:** Diagrams and illustrations will be added in a future update.

---

## Background

The designer of AnimaWorks is a psychiatrist with over 30 years of programming experience. AnimaWorks' memory system, autonomic mechanisms, and execution architecture are **intentionally** mapped to the structure of the human brain, grounded in clinical neuroscience. This is not merely a metaphor -- it is an attempt to reuse the brain's information-processing architecture as a design pattern.

In psychiatric practice, one routinely observes dysfunctions of the brain's various subsystems: memory disorders, attention disorders, executive function disorders, and more. Knowing what happens when each subsystem is impaired made it possible to identify the subsystems an AI agent requires and to design a clear separation of their respective roles.

---

## Overall Mapping

### Neocortex -- The LLM

| LLM Function | Brain Region | Description |
|---|---|---|
| Reasoning & decision-making | Prefrontal cortex (PFC) | Executive function. Receives memories injected by priming and makes judgments |
| Language comprehension | Wernicke's area (temporal lobe) | Semantic understanding of input messages |
| Language production | Broca's area (frontal lobe) | Generation of response text |
| Pre-trained knowledge | Crystallized patterns in temporal cortex | World knowledge baked into LLM weights. A separate system from file-based memory -- "innate intelligence" |
| Transformer Attention | Parietal association cortex + PFC selective attention | Allocation of attention to relevant information within the context |

The LLM in its entirety corresponds to the **neocortex** as a whole. However, in AnimaWorks' design, because the framework handles subcortical functions (memory consolidation, forgetting, arousal maintenance), the role left to the LLM is effectively distilled into the **conscious processing of the prefrontal cortex (PFC)**.

As memory.md states:

> The agent (LLM) is "the one who thinks," not "the administrator of its own brain."

### Duality of Pre-trained Knowledge and File-based Memory

The knowledge baked into the LLM's pre-trained weights and AnimaWorks' file-based memory are **separate systems**. In the human brain as well, patterns crystallized in the cerebral cortex (implicit knowledge / crystallized intelligence) and episodic memory via the hippocampus function as independent systems.

| Type of Knowledge | Human Brain | AnimaWorks |
|---|---|---|
| Innate intelligence | Crystallized intelligence (cortical patterns) | LLM pre-trained weights |
| Experientially acquired knowledge | Fluid intelligence + episodic memory | File-based memory (episodes/, knowledge/, procedures/) |

This distinction aligns with the "imperfect individual" design philosophy described in vision.md. Precisely because pre-trained knowledge alone is insufficient, an experience-based memory system is necessary.

---

### Memory System -- Hippocampus, Cerebral Cortex, & Basal Ganglia

| Human Memory | Brain Region | AnimaWorks Implementation | Characteristics |
|---|---|---|---|
| **Working memory** | Prefrontal cortex | LLM context window | Capacity-limited. Temporary holding of "what is currently being thought about" |
| **Episodic memory** | Hippocampus -> neocortex | `episodes/` | Chronological record of "when and what happened" |
| **Semantic memory** | Temporal cortex | `knowledge/` | Lessons and knowledge decoupled from context |
| **Procedural memory** | Basal ganglia, cerebellum | `procedures/`, `skills/` | "How to do it." Strengthened through repetition |
| **Person memory** | Fusiform gyrus, temporal pole | `shared/users/` | Automatic recall of "who is this person" |

### Internal Structure of Working Memory -- Baddeley's Model

Based on Baddeley (2000):

| Baddeley's Component | Function | AnimaWorks Implementation |
|---|---|---|
| **Central executive** | Attentional control; orchestration of retrieval from long-term memory | Agent orchestrator |
| **Episodic buffer** | Integration of multiple sources into a unified representation | Context assembly layer (priming results + conversation history) |
| **Phonological loop** | Temporary holding of verbal information | Text buffer (recent conversation turns) |

Following Cowan (2005), working memory is understood as a "spotlight on activated long-term memory." The context window is not an independent store, but rather the portion of long-term memory that currently has attention directed toward it.

---

### Memory Recall -- Dual Pathways

| Recall Pathway | Brain Process | AnimaWorks Implementation |
|---|---|---|
| **Automatic recall** | Pattern completion by the CA3 auto-associative network of the hippocampus. Unconscious, fast (250-500 ms), unsuppressible | Priming layer (4-channel parallel search) |
| **Deliberate recall** | Strategic search by the prefrontal cortex (PFC). Conscious, slow | `search_memory` / `read_memory_file` tools |

### Spreading Activation -- Collins & Loftus (1975)

| Search Signal | Brain Counterpart | AnimaWorks Implementation |
|---|---|---|
| Semantic neighborhood discovery | Spreading activation among concept nodes | Dense vector similarity search (ChromaDB) |
| Prioritization of recent memories | Recency effect | Time-decay function (half-life: 30 days) |
| Strengthening of frequently used memories | Hebb's rule / Long-term potentiation (LTP) | Access frequency boost |
| Multi-hop association | Propagation through associative networks | Knowledge graph + Personalized PageRank |

---

### Memory Consolidation -- Sleep and Integration

| AnimaWorks | Brain Process | Description |
|---|---|---|
| **Immediate encoding** (session boundary) | Hippocampal rapid one-shot encoding | At conversation end, a differential summary is recorded in episodes/ |
| **Daily consolidation** (midnight cron) | NREM slow-wave -- spindle -- ripple cascade | Knowledge extraction and validation from episodes/ to knowledge/ |
| **Weekly integration** | Neocortical long-term consolidation | Deduplication and merging of knowledge/, pattern distillation |
| **NLI + LLM validation** | Hippocampal pattern separation | Hallucination elimination. Consistency verification between episodes and extracted knowledge |
| **Prediction-error-based reconsolidation** | Reconsolidation theory, Nader et al. (2000) | Updating memory when new information contradicts existing memory |

---

### Forgetting -- Synaptic Homeostasis

Based on the synaptic homeostasis hypothesis of Tononi & Cirelli (2003):

| AnimaWorks | Brain Process | Description |
|---|---|---|
| **Daily downscaling** | Synaptic downscaling during NREM sleep | Marking of low-activity chunks |
| **Neurogenesis-inspired reorganization** | Memory circuit reorganization via neurogenesis in the hippocampal dentate gyrus | LLM-driven merging of low-activity + similar chunks |
| **Complete forgetting** | Elimination of sub-threshold synapses | Archive -> deletion |
| **Forgetting resistance** (procedures, skills) | Procedural memory in basal ganglia is resistant to forgetting | Protected by version >= 3 or protected: true |

---

### Arousal & Autonomic Mechanisms

| AnimaWorks | Brain Region | Description |
|---|---|---|
| **Heartbeat** (periodic patrol) | **Reticular activating system (ARAS)** | Maintenance of arousal. Does not specify the content of consciousness; provides the precondition for consciousness. Fires rhythmically -- without it, dormancy (coma) ensues |
| **Cron** (scheduled tasks) | Hypothalamic circadian rhythm (SCN) | Time-based periodic action triggers. Sleep-wake cycle, daily/weekly/monthly biorhythms |
| **ProcessSupervisor** | Autonomic nervous system | Manages process life and death. Operates outside consciousness, handling startup, monitoring, and restart of each Anima |
| **Unix Domain Socket IPC** | Nerve fiber bundles (white matter tracts) | Physical communication pathways between Anima processes |
| **Messenger** | Synaptic transmission | Sending and receiving messages. Connecting encapsulated individuals through text |

#### Heartbeat = Reticular Activating System (ARAS) in Detail

The Ascending Reticular Activating System (ARAS) projects from the brainstem's reticular formation through the thalamus to the entire cerebral cortex, maintaining the state of arousal. Its characteristics correspond to AnimaWorks' heartbeat as follows:

| ARAS Characteristic | Heartbeat Characteristic |
|---|---|
| Maintains arousal (does not specify the content of consciousness) | Periodically activates the Anima (what to think about is delegated to heartbeat.md) |
| Fires automatically and rhythmically | Executes automatically at configured intervals |
| Functional cessation leads to coma | Without heartbeat, the Anima remains dormant unless externally stimulated (by messages) |
| A precondition for consciousness, not consciousness itself | A precondition for autonomous action, not the judgment itself |
| Arousal level fluctuates with sensory input | Activates immediately upon receiving a message (even outside the heartbeat cycle) |

---

### Organizational Structure -- The Social Brain

| AnimaWorks | Brain / Psychology Counterpart | Description |
|---|---|---|
| **Supervisor-subordinate hierarchy** | Neural basis of social hierarchy (PFC-amygdala circuit) | Flow of instructions and reports |
| **Encapsulation (internal state invisible)** | Theory of Mind (ToM) | The internal state of others can only be inferred |
| **Message-based communication** | Linguistic communication | Connected only through text. No shared memory or direct references |
| **identity.md (personality)** | Personality (stable PFC-limbic patterns) | Immutable baseline. Foundation for judgment |
| **injection.md (role)** | Social role / occupational identity | Mutable. Behavioral guidelines within the organization |

---

## Neuroscientific Rationale for Design Principles

### Why Context Window Limits Are a "Feature"

Human working memory capacity is limited to 4 +/- 1 chunks (Cowan, 2001). This is not a defect but an **evolutionary adaptation that enforces selective attention, thereby ensuring quality of judgment**. If all memories surfaced in consciousness simultaneously, relevant information could not be selected, and judgment would deteriorate.

AnimaWorks adopts this principle as a "design feature." Only the necessary information is recalled through priming, enabling judgment within a clean context.

### Why Forgetting Is Necessary

Synaptic downscaling during sleep globally weakens synapses that were strengthened during wakefulness, maintaining the signal-to-noise ratio (SNR). Without forgetting, accumulated memories become noise, degrading search precision.

AnimaWorks' active forgetting reproduces this biological mechanism, maintaining vector search accuracy over the long term.

### Why Collaboration Among "Imperfect Individuals" Is More Robust Than Omniscient Individuals

Human organizations function because each member makes judgments with limited perspective and memory, communicating imperfect information in their own words (vision.md). This aligns with Cognitive Load Theory (Sweller, 1988) and Distributed Cognition (Hutchins, 1995).

---

## Summary

AnimaWorks is a design born at the intersection of a psychiatrist's clinical knowledge and engineering experience. The brain's information-processing architecture is a **universal design pattern** that can be reused independently of its biological substrate (neurons), and AnimaWorks is a system that demonstrates this.

By mapping the LLM to the neocortex, the memory system to the hippocampal-cortical system, heartbeat to the reticular activating system, and forgetting to synaptic homeostasis -- and by having these operate in an integrated manner -- AnimaWorks realizes "an entity that autonomously thinks, learns, forgets, and collaborates."
