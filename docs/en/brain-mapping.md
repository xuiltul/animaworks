<!-- 自動翻訳ファイル・編集禁止 (AUTO-TRANSLATED, DO NOT EDIT). 正本: docs/ja/brain-mapping.md -->
<!-- i18n: source-sha256=7dcc78847528444765320f4b4c5136301eba11e1e31df7999f11faa067f64b19 generated=2026-10-31 engine=luna model=gpt-6-luna-2026-09-22 translator=2 -->

> Confirmed commit: 193a5e72

# Correspondence with Neuroscience

## Background

AnimaWorks uses the differences in memory, attention, and executive function observed in psychiatric clinical practice, along with the modular separation of software engineering, as design cues. Rather than imitating the brain directly, it references findings from neuroscience as a design correspondence for dividing information-processing roles.

## Overall Mapping

| AnimaWorks Concept | Corresponding Neuroscience / Cognitive Science | Key Points of Correspondence |
|---|---|---|
| Reasoning and language processing by LLM | Executive function of the cerebral neocortex, especially the prefrontal cortex | Uses recalled information to make judgments and construct responses. |
| LLM pretrained knowledge | General knowledge patterns formed in the cerebral cortex | A separate system from file-based memory accumulated through individual experience. |
| Context | Working memory | Temporarily holds the current focus of attention. See [Memory System](memory/index.md) for details. |
| Episodes, knowledge, and procedures | Memory functions of the hippocampal system, cerebral cortex, basal ganglia, etc. | Holds distinct roles—events, meanings, and procedures—separately. |
| Automatic recall and explicit search | Cue-based recall and prefrontal exploration | Combines automatic recall from context with intentional search when needed. |
| Heartbeat and scheduled execution | Arousal maintenance and circadian rhythm | Treats periodic startup and time-specified execution as a foundation separate from individual judgment. |
| Collaboration of multiple Anima | Distributed cognition and social cognition | Does not aggregate individual internal states into shared memory but collaborates through communication and role division. |

## Automatic Recall and Selective Attention

Automatic recall is a mechanism that presents relevant memories into the context in response to cues, analogous to associative reactivation in the brain. The implementation defaults to `compact` and narrows down to the information needed for the current execution. See [Automatic Recall](memory/priming.md) for details on channels and search conditions.

## Neuroscientific Basis of Design Principles

### Context Limitation

Working memory has capacity constraints (Cowan, 2001). A limited context is not merely a drawback but a design condition that directs attention to relevant information. AnimaWorks recalls necessary cues and does not present the entire long-term memory each time.

### Consolidation and Forgetting

Research on memory consolidation during sleep and synaptic homeostasis (Tononi & Cirelli, 2003) shows that both retention and selection are involved in memory function. AnimaWorks also provides opportunities to review usage and content, not just long-term storage of records. See [Memory Consolidation and Forgetting](memory/consolidation.md) for the current process.

### Collaboration of Incomplete Individuals

The design that limits each Anima's knowledge and perspective and brings information together through messages is consistent with distributed cognition (Hutchins, 1995) and the concept of cognitive load (Sweller, 1988). Rather than aggregating all information and judgment into a single agent, the system is composed of collaboration among individuals with distinct roles.

## How to Read This Section

The correspondences here are mappings to explain the design and do not claim that software functions and brain regions are biologically identical. See [Memory System](memory/index.md) for the implemented memory structure and [Execution Mode](architecture/execution.md) for execution modes.
