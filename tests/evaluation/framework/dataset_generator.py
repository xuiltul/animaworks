# AnimaWorks - Digital Anima Framework
# Copyright (C) 2026 AnimaWorks Authors
# SPDX-License-Identifier: Apache-2.0
"""
Dataset generator for memory performance evaluation.

Generates realistic memory bases and conversation scenarios for testing.
"""
from __future__ import annotations

import json
import random
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from typing import Literal

from .schemas import (
    ConversationTurn,
    MemoryBase,
    MemoryFile,
    Scenario,
    ScenarioTypeConfig,
    SizeConfig
)


# ── Dataset Generator ───────────────────────────────────────────────────────


class DatasetGenerator:
    """
    Generates memory bases and scenarios for evaluation.

    Uses LLM (via litellm) to create high-quality, realistic content.
    Falls back to template-based generation if LLM is unavailable.
    """

    def __init__(
        self,
        output_dir: Path,
        use_llm: bool = True,
        model: str = "anthropic/claude-sonnet-4-20250514"
    ):
        """
        Initialize dataset generator.

        Args:
            output_dir: Root directory for generated datasets
            use_llm: Whether to use LLM for content generation
            model: LLM model to use (litellm format)
        """
        self.output_dir = Path(output_dir)
        self.use_llm = use_llm
        self.model = model

        # Import litellm only if needed
        if self.use_llm:
            try:
                import litellm
                self.litellm = litellm
            except ImportError:
                print("Warning: litellm not available, falling back to templates")
                self.use_llm = False

    # ── Memory Base Generation ───────────────────────────────────────────────

    def generate_memory_base(
        self,
        domain: Literal["business", "tech_support", "education"],
        size: Literal["small", "medium", "large"]
    ) -> MemoryBase:
        """
        Generate complete memory base for a domain and size.

        Args:
            domain: Domain category
            size: Dataset size (small/medium/large)

        Returns:
            Complete memory base with all files
        """
        size_config = SizeConfig.get_config(size)

        print(f"Generating {size} {domain} memory base...")
        print(f"  - {size_config.knowledge_count} knowledge files")
        print(f"  - {size_config.episode_count} episode files")
        print(f"  - {size_config.skill_count} skill files")

        # Create output directories
        base_dir = self.output_dir / domain / size
        knowledge_dir = base_dir / "knowledge"
        episodes_dir = base_dir / "episodes"
        skills_dir = base_dir / "skills"

        for dir_path in [knowledge_dir, episodes_dir, skills_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Generate files
        knowledge_files = self._generate_knowledge_files(
            domain=domain,
            count=size_config.knowledge_count,
            output_dir=knowledge_dir
        )

        episode_files = self._generate_episode_files(
            domain=domain,
            count=size_config.episode_count,
            output_dir=episodes_dir
        )

        skill_files = self._generate_skill_files(
            domain=domain,
            count=size_config.skill_count,
            output_dir=skills_dir
        )

        memory_base = MemoryBase(
            domain=domain,
            size=size,
            knowledge_files=knowledge_files,
            episode_files=episode_files,
            skill_files=skill_files
        )

        print(f"✓ Generated {memory_base.total_files} files ({memory_base.total_tokens:,} tokens)")

        return memory_base

    # ── Knowledge File Generation ────────────────────────────────────────────

    def _generate_knowledge_files(
        self,
        domain: str,
        count: int,
        output_dir: Path
    ) -> list[MemoryFile]:
        """Generate knowledge files for a domain."""
        files = []

        topics = self._get_knowledge_topics(domain)

        for i in range(count):
            topic = topics[i % len(topics)]
            filename = f"{topic.lower().replace(' ', '_')}_{i:04d}.md"
            file_path = output_dir / filename

            content = self.generate_knowledge_file(domain, topic, i)
            tokens = self._estimate_tokens(content)

            file_path.write_text(content, encoding="utf-8")

            files.append(MemoryFile(
                path=file_path,
                content=content,
                tokens=tokens,
                metadata={"topic": topic, "index": i}
            ))

        return files

    def generate_knowledge_file(
        self,
        domain: str,
        topic: str,
        index: int = 0
    ) -> str:
        """
        Generate a single knowledge file (500-5000 chars, avg 1500).

        Args:
            domain: Domain category
            topic: Topic for this knowledge file
            index: Index for variation

        Returns:
            File content in markdown
        """
        if self.use_llm:
            return self._generate_knowledge_with_llm(domain, topic, index)
        else:
            return self._generate_knowledge_template(domain, topic, index)

    def _generate_knowledge_with_llm(
        self,
        domain: str,
        topic: str,
        index: int
    ) -> str:
        """Generate knowledge file using LLM."""
        prompt = f"""Generate a knowledge document for a {domain} assistant.

Topic: {topic}
Target length: 1000-2000 characters
Format: Markdown

Create realistic, detailed content that would be useful for an AI assistant.
Include specific facts, procedures, or guidelines.
Use headers, lists, and formatting as appropriate.

Write in Japanese if the domain is business, otherwise use English."""

        try:
            response = self.litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=1000
            )
            content = response.choices[0].message.content
            return f"# {topic}\n\n{content}"
        except Exception as e:
            print(f"Warning: LLM generation failed ({e}), using template")
            return self._generate_knowledge_template(domain, topic, index)

    def _generate_knowledge_template(
        self,
        domain: str,
        topic: str,
        index: int
    ) -> str:
        """Generate knowledge file using template (fallback)."""
        return f"""# {topic}

## Overview

This document contains information about {topic.lower()} in the context of {domain}.

## Key Points

- Point 1: Important information about {topic}
- Point 2: Additional details and context
- Point 3: Best practices and guidelines
- Point 4: Common scenarios and examples

## Details

Lorem ipsum dolor sit amet, consectetur adipiscing elit. This is placeholder
content for knowledge file {index}. In a real scenario, this would contain
detailed information about {topic} relevant to {domain}.

## References

- Related Topic A
- Related Topic B
- External Resource C

---
*Last updated: {datetime.now().strftime('%Y-%m-%d')}*
"""

    def _get_knowledge_topics(self, domain: str) -> list[str]:
        """Get topic list for knowledge files."""
        topics = {
            "business": [
                "Company Vision", "Client Management", "Project Guidelines",
                "Communication Protocol", "Meeting Templates", "Report Format",
                "Budget Planning", "Risk Management", "Quality Standards",
                "Team Structure", "Onboarding Process", "Performance Review"
            ],
            "tech_support": [
                "Product Specifications", "Troubleshooting Guide", "Error Codes",
                "Installation Process", "Configuration Guide", "API Reference",
                "Security Guidelines", "Backup Procedures", "System Requirements",
                "Common Issues", "Performance Tuning", "Update Protocol"
            ],
            "education": [
                "Learning Theory", "Curriculum Design", "Assessment Methods",
                "Student Engagement", "Lesson Planning", "Differentiation",
                "Technology Integration", "Classroom Management", "Feedback Strategies",
                "Parent Communication", "Progress Tracking", "Special Needs Support"
            ]
        }
        return topics.get(domain, ["General Topic"])

    # ── Episode File Generation ──────────────────────────────────────────────

    def _generate_episode_files(
        self,
        domain: str,
        count: int,
        output_dir: Path
    ) -> list[MemoryFile]:
        """Generate episode files for a domain."""
        files = []

        # Generate episodes for consecutive days
        start_date = datetime.now() - timedelta(days=count)

        for i in range(count):
            date = start_date + timedelta(days=i)
            date_str = date.strftime("%Y-%m-%d")
            filename = f"{date_str}.md"
            file_path = output_dir / filename

            content = self.generate_episode_file(domain, date_str, i)
            tokens = self._estimate_tokens(content)

            file_path.write_text(content, encoding="utf-8")

            files.append(MemoryFile(
                path=file_path,
                content=content,
                tokens=tokens,
                metadata={"date": date_str, "index": i}
            ))

        return files

    def generate_episode_file(
        self,
        domain: str,
        date: str,
        index: int = 0
    ) -> str:
        """
        Generate a single episode file (1000-3000 chars, avg 2000).

        Args:
            domain: Domain category
            date: Date string (YYYY-MM-DD)
            index: Index for variation

        Returns:
            File content in markdown
        """
        if self.use_llm:
            return self._generate_episode_with_llm(domain, date, index)
        else:
            return self._generate_episode_template(domain, date, index)

    def _generate_episode_with_llm(
        self,
        domain: str,
        date: str,
        index: int
    ) -> str:
        """Generate episode file using LLM."""
        prompt = f"""Generate a daily episode log for a {domain} assistant.

Date: {date}
Target length: 1500-2500 characters
Format: Markdown with timeline

Create realistic activities that happened during this day.
Include specific events, conversations, tasks completed, and decisions made.
Use time stamps (e.g., ## 09:00 - Meeting with Client)

Write in Japanese if the domain is business, otherwise use English."""

        try:
            response = self.litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=1500
            )
            content = response.choices[0].message.content
            return f"# Daily Episode - {date}\n\n{content}"
        except Exception as e:
            print(f"Warning: LLM generation failed ({e}), using template")
            return self._generate_episode_template(domain, date, index)

    def _generate_episode_template(
        self,
        domain: str,
        date: str,
        index: int
    ) -> str:
        """Generate episode file using template (fallback)."""
        return f"""# Daily Episode - {date}

## 09:00 - Morning Review

Started the day by reviewing pending tasks and priorities. Checked messages
and planned the day's activities for {domain} work.

## 10:30 - Main Activity

Completed primary task related to ongoing project. Made progress on key
deliverables and coordinated with stakeholders.

## 14:00 - Afternoon Session

Addressed several requests and questions. Provided support and guidance
on various topics. Documented findings and updates.

## 16:00 - Wrap-up

Summarized day's accomplishments. Updated task list and prepared for
tomorrow's priorities.

---
*Episode #{index}*
"""

    # ── Skill File Generation ────────────────────────────────────────────────

    def _generate_skill_files(
        self,
        domain: str,
        count: int,
        output_dir: Path
    ) -> list[MemoryFile]:
        """Generate skill files for a domain."""
        files = []

        skills = self._get_skill_names(domain)

        for i in range(count):
            skill_name = skills[i % len(skills)]
            filename = f"{skill_name.lower().replace(' ', '_')}_{i:04d}.md"
            file_path = output_dir / filename

            content = self.generate_skill_file(domain, skill_name, i)
            tokens = self._estimate_tokens(content)

            file_path.write_text(content, encoding="utf-8")

            files.append(MemoryFile(
                path=file_path,
                content=content,
                tokens=tokens,
                metadata={"skill": skill_name, "index": i}
            ))

        return files

    def generate_skill_file(
        self,
        domain: str,
        skill_name: str,
        index: int = 0
    ) -> str:
        """
        Generate a single skill file (300-1000 chars, avg 500).

        Args:
            domain: Domain category
            skill_name: Name of the skill
            index: Index for variation

        Returns:
            File content in markdown
        """
        if self.use_llm:
            return self._generate_skill_with_llm(domain, skill_name, index)
        else:
            return self._generate_skill_template(domain, skill_name, index)

    def _generate_skill_with_llm(
        self,
        domain: str,
        skill_name: str,
        index: int
    ) -> str:
        """Generate skill file using LLM."""
        prompt = f"""Generate a skill/procedure document for a {domain} assistant.

Skill: {skill_name}
Target length: 400-700 characters
Format: Step-by-step guide in Markdown

Create a concise, actionable procedure that the assistant can follow.
Include numbered steps and key considerations.

Write in Japanese if the domain is business, otherwise use English."""

        try:
            response = self.litellm.completion(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=500
            )
            content = response.choices[0].message.content
            return f"# Skill: {skill_name}\n\n{content}"
        except Exception as e:
            print(f"Warning: LLM generation failed ({e}), using template")
            return self._generate_skill_template(domain, skill_name, index)

    def _generate_skill_template(
        self,
        domain: str,
        skill_name: str,
        index: int
    ) -> str:
        """Generate skill file using template (fallback)."""
        return f"""# Skill: {skill_name}

## Steps

1. Prepare necessary information and context
2. Execute the primary action for {skill_name}
3. Verify results and quality
4. Document outcomes and learnings

## Tips

- Consider edge cases
- Maintain consistency
- Follow best practices

---
*Skill #{index} for {domain}*
"""

    def _get_skill_names(self, domain: str) -> list[str]:
        """Get skill names for a domain."""
        skills = {
            "business": [
                "Report Creation", "Meeting Scheduling", "Email Drafting",
                "Budget Analysis", "Risk Assessment", "Client Communication"
            ],
            "tech_support": [
                "Log Analysis", "Debugging", "Issue Triage",
                "System Diagnostics", "Patch Deployment", "User Training"
            ],
            "education": [
                "Curriculum Design", "Assessment Creation", "Feedback Writing",
                "Lesson Planning", "Student Motivation", "Parent Communication"
            ]
        }
        return skills.get(domain, ["General Skill"])

    # ── Scenario Generation ──────────────────────────────────────────────────

    def generate_scenarios(
        self,
        domain: Literal["business", "tech_support", "education"],
        memory_base: MemoryBase,
        total_count: int = 50
    ) -> list[Scenario]:
        """
        Generate conversation scenarios for evaluation.

        Args:
            domain: Domain category
            memory_base: Memory base to reference in scenarios
            total_count: Total number of scenarios to generate

        Returns:
            List of scenarios with different types
        """
        config = ScenarioTypeConfig()
        type_counts = config.get_counts(total_count)

        scenarios = []

        # Generate each type
        for scenario_type, count in type_counts.items():
            for i in range(count):
                scenario = self._generate_scenario(
                    domain=domain,
                    scenario_type=scenario_type,  # type: ignore
                    memory_base=memory_base,
                    index=i
                )
                scenarios.append(scenario)

        return scenarios

    def _generate_scenario(
        self,
        domain: str,
        scenario_type: Literal["factual", "episodic", "multihop", "long"],
        memory_base: MemoryBase,
        index: int
    ) -> Scenario:
        """Generate a single scenario."""
        scenario_id = f"{domain}_{scenario_type}_{index:03d}"

        # Select relevant memory files for this scenario
        relevant_files = self._select_relevant_memories(memory_base, scenario_type)

        # Generate turns based on scenario type
        if scenario_type == "factual":
            turns = self._generate_factual_turns(domain, relevant_files)
        elif scenario_type == "episodic":
            turns = self._generate_episodic_turns(domain, relevant_files)
        elif scenario_type == "multihop":
            turns = self._generate_multihop_turns(domain, relevant_files)
        else:  # long
            turns = self._generate_long_conversation_turns(domain, relevant_files)

        return Scenario(
            scenario_id=scenario_id,
            scenario_type=scenario_type,
            domain=domain,  # type: ignore
            turns=turns,
            metadata={"generated_at": datetime.now().isoformat()}
        )

    def _select_relevant_memories(
        self,
        memory_base: MemoryBase,
        scenario_type: str
    ) -> list[MemoryFile]:
        """Select relevant memories for a scenario type."""
        if scenario_type == "factual":
            # Focus on knowledge files
            return random.sample(memory_base.knowledge_files, min(2, len(memory_base.knowledge_files)))
        elif scenario_type == "episodic":
            # Focus on episode files
            return random.sample(memory_base.episode_files, min(2, len(memory_base.episode_files)))
        elif scenario_type == "multihop":
            # Mix of knowledge and episodes
            files = (
                random.sample(memory_base.knowledge_files, min(2, len(memory_base.knowledge_files))) +
                random.sample(memory_base.episode_files, min(1, len(memory_base.episode_files)))
            )
            return files
        else:  # long
            # Mix of all types
            return random.sample(memory_base.all_files, min(5, len(memory_base.all_files)))

    def _generate_factual_turns(
        self,
        domain: str,
        relevant_files: list[MemoryFile]
    ) -> list[ConversationTurn]:
        """Generate turns for factual recall scenario (5 turns)."""
        turns = []

        for i in range(5):
            if i < len(relevant_files):
                file = relevant_files[i]
                message = f"What information do you have about {file.metadata.get('topic', 'this topic')}?"
                relevant_paths = [file.path]
            else:
                message = f"Can you summarize what we discussed in turn {i}?"
                relevant_paths = [f.path for f in relevant_files]

            turns.append(ConversationTurn(
                message=message,
                relevant_memories=relevant_paths
            ))

        return turns

    def _generate_episodic_turns(
        self,
        domain: str,
        relevant_files: list[MemoryFile]
    ) -> list[ConversationTurn]:
        """Generate turns for episodic recall scenario (5 turns)."""
        turns = []

        for i in range(5):
            if i < len(relevant_files):
                file = relevant_files[i]
                date = file.metadata.get('date', 'that day')
                message = f"What happened on {date}?"
                relevant_paths = [file.path]
            else:
                message = f"What were the key events in recent days?"
                relevant_paths = [f.path for f in relevant_files]

            turns.append(ConversationTurn(
                message=message,
                relevant_memories=relevant_paths
            ))

        return turns

    def _generate_multihop_turns(
        self,
        domain: str,
        relevant_files: list[MemoryFile]
    ) -> list[ConversationTurn]:
        """Generate turns for multihop reasoning scenario (10 turns)."""
        turns = []

        # First half: gather information
        for i in range(5):
            if i < len(relevant_files):
                file = relevant_files[i]
                topic = file.metadata.get('topic', file.metadata.get('date', 'this'))
                message = f"Tell me about {topic}."
                relevant_paths = [file.path]
            else:
                message = f"What else can you tell me about this topic?"
                relevant_paths = [f.path for f in relevant_files[:i]]

            turns.append(ConversationTurn(
                message=message,
                relevant_memories=relevant_paths
            ))

        # Second half: reasoning
        for i in range(5, 10):
            message = f"Based on what we discussed, what conclusions can we draw?"
            relevant_paths = [f.path for f in relevant_files]

            turns.append(ConversationTurn(
                message=message,
                relevant_memories=relevant_paths
            ))

        return turns

    def _generate_long_conversation_turns(
        self,
        domain: str,
        relevant_files: list[MemoryFile]
    ) -> list[ConversationTurn]:
        """Generate turns for long conversation scenario (20 turns)."""
        turns = []

        for i in range(20):
            file = relevant_files[i % len(relevant_files)]
            topic = file.metadata.get('topic', file.metadata.get('date', file.metadata.get('skill', 'this')))
            message = f"Let's discuss {topic} in more detail (turn {i+1})."
            relevant_paths = [file.path]

            turns.append(ConversationTurn(
                message=message,
                relevant_memories=relevant_paths
            ))

        return turns

    # ── Scenario Storage ─────────────────────────────────────────────────────

    def save_scenarios(
        self,
        scenarios: list[Scenario],
        output_dir: Path
    ) -> list[Path]:
        """
        Save scenarios to YAML files.

        Args:
            scenarios: List of scenarios to save
            output_dir: Directory to save scenario files

        Returns:
            List of paths to saved files
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths = []

        for scenario in scenarios:
            filename = f"{scenario.scenario_id}.yaml"
            file_path = output_dir / filename

            data = {
                "scenario_id": scenario.scenario_id,
                "type": scenario.scenario_type,
                "domain": scenario.domain,
                "metadata": scenario.metadata,
                "turns": [
                    {
                        "message": turn.message,
                        "relevant_memories": [str(p) for p in turn.relevant_memories],
                        "expected_answer": turn.expected_answer,
                        "metadata": turn.metadata
                    }
                    for turn in scenario.turns
                ]
            }

            with open(file_path, "w", encoding="utf-8") as f:
                yaml.dump(data, f, allow_unicode=True, default_flow_style=False)

            saved_paths.append(file_path)

        return saved_paths

    # ── Utilities ────────────────────────────────────────────────────────────

    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count (rough approximation: 1 token ≈ 4 chars)."""
        return len(text) // 4
