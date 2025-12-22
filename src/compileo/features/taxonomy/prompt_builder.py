"""
Prompt builder for taxonomy generation.
"""

import json
from typing import Dict, Any, Optional, List


class TaxonomyPromptBuilder:
    """
    Builder for creating prompts for taxonomy generation and extension.
    """

    @staticmethod
    def build_dynamic_taxonomy_prompt(content: str) -> str:
        """
        Create a dynamic, parameter-free prompt for taxonomy generation using role playing and chain of thought.

        Args:
            content: Content to analyze for taxonomy generation

        Returns:
            Dynamic prompt combining role playing with chain of thought reasoning
        """
        prompt = f"""You are a master taxonomist with decades of experience organizing complex knowledge domains into hierarchical structures. When analyzing content for categorization, follow this systematic reasoning process:

**Step 1: Content Analysis**
Identify the core themes, concepts, and patterns that emerge naturally from the content. Look for recurring ideas, relationships, and organizational principles.

**Step 2: Hierarchical Structure**
Determine how these themes relate to each other. Consider which concepts are foundational, which are subcategories, and how they form a logical tree structure.

**Step 3: Category Refinement**
Refine the categories to ensure they are mutually exclusive where possible, have clear boundaries, and represent meaningful distinctions in the content.

**Step 4: Confidence Assessment**
Assign confidence levels based on how strongly each category is supported by evidence in the content.

**Content to Analyze:**
{content}

Create a hierarchical taxonomy that captures the essential structure of this content. Use clear, descriptive category names and logical parent-child relationships. Include confidence scores (0.0-1.0) for each category based on content evidence.

Return ONLY a valid JSON object with the taxonomy structure, no additional text or explanations."""

        return prompt

    @staticmethod
    def build_taxonomy_prompt(
        content: str,
        domain: str,
        depth: Optional[int],
        category_limits: Optional[List[int]] = None,
        specificity_level: int = 1
    ) -> str:
        """
        Create a domain-specific prompt for taxonomy generation.

        Args:
            content: Sampled content from chunks
            domain: Knowledge domain
            depth: Maximum hierarchy depth
            category_limits: Optional limits for categories per level
            specificity_level: Level of specificity for categories

        Returns:
            Complete prompt for Grok
        """
        domain_context = TaxonomyPromptBuilder._get_domain_context(domain)
        specificity_instruction = TaxonomyPromptBuilder._get_specificity_instruction(specificity_level)

        # Add category limits to the prompt if provided
        limits_instruction = ""
        if category_limits:
            limits_instruction = f"\n**Category Limits:**\n"
            for i, limit in enumerate(category_limits, 1):
                limits_instruction += f"- Level {i}: Maximum {limit} categories\n"
            limits_instruction += "\n"

        prompt = f"""You are an expert taxonomist specializing in {domain} knowledge organization. Your task is to analyze the provided content and create a hierarchical taxonomy that captures the key concepts, categories, and relationships within the {domain} domain.

**Domain Context:** {domain_context}

**Task Requirements:**
1. Analyze the content to identify main themes, concepts, and categories
{f"2. Create a hierarchical taxonomy with maximum depth of {depth} levels" if depth else "2. Create a hierarchical taxonomy that best captures the content structure (no fixed depth limit)"}
3. Ensure categories are specific to the {domain} domain
4. Use clear, descriptive names for categories
5. Include confidence scores (0.0-1.0) for each category based on content evidence
6. Make categories mutually exclusive where possible
7. Focus on categories that would be useful for content classification and organization{specificity_instruction}{limits_instruction}

**Content to Analyze:**
{content}

**Output Format:**
Return a valid JSON object with this exact structure:
{{
  "name": "Root Category Name",
  "description": "Brief description of the root category",
  "children": [
    {{
      "name": "Level 1 Category",
      "description": "Description of this category",
      "confidence_threshold": 0.8,
      "children": [
        {{
          "name": "Level 2 Category",
          "description": "Description of subcategory",
          "confidence_threshold": 0.7,
          "children": [...]
        }}
      ]
    }}
  ]
}}

**Important:**
- Return ONLY the JSON object, no other text
- Ensure the JSON is valid and properly formatted
{f"- Maximum depth: {depth} levels" if depth else "- Determine the optimal depth based on content complexity"}
- Include confidence_threshold for each category (0.0-1.0)
- Make category names specific and actionable"""

        return prompt

    @staticmethod
    def build_dynamic_taxonomy_extension_prompt(
        existing_taxonomy: Dict[str, Any],
        content: str
    ) -> str:
        """
        Create a dynamic, parameter-free prompt for taxonomy extension using role playing and chain of thought.

        Args:
            existing_taxonomy: The existing taxonomy to extend
            content: Content to analyze for extension

        Returns:
            Dynamic prompt for taxonomy extension
        """
        # Convert existing taxonomy to readable format
        existing_taxonomy_str = json.dumps(existing_taxonomy, indent=2)

        prompt = f"""You are a master taxonomist with decades of experience expanding and refining knowledge organization systems. When extending an existing taxonomy, follow this systematic reasoning process:

**Step 1: Structure Analysis**
Examine the existing taxonomy to understand its current organization, depth, and coverage. Identify areas where additional subcategories would be most valuable.

**Step 2: Content Integration**
Analyze the new content to find concepts and relationships that aren't adequately represented in the current taxonomy structure.

**Step 3: Extension Planning**
Determine where and how to add new levels or branches to the taxonomy. Consider logical hierarchy and maintain consistency with existing categories.

**Step 4: Category Development**
Create new subcategories that fit naturally within the existing structure while capturing the nuances present in the new content.

**Existing Taxonomy to Extend:**
{existing_taxonomy_str}

**New Content to Integrate:**
{content}

Extend the taxonomy by adding appropriate subcategories and deeper levels where needed. Maintain the existing structure while incorporating new concepts discovered in the content. Include confidence scores for new categories based on content evidence.

Return ONLY a valid JSON object with the extended taxonomy structure, no additional text or explanations."""

        return prompt

    @staticmethod
    def build_taxonomy_extension_prompt(
        existing_taxonomy: Dict[str, Any],
        content: str,
        additional_depth: Optional[int],
        domain: str,
        category_limits: Optional[List[int]] = None,
        specificity_level: int = 1
    ) -> str:
        """
        Create a prompt for extending an existing taxonomy.

        Args:
            existing_taxonomy: The existing taxonomy to extend
            content: Sampled content from chunks
            additional_depth: Number of additional levels to add
            domain: Knowledge domain
            category_limits: Optional limits for categories per level
            specificity_level: Level of specificity for new categories

        Returns:
            Complete prompt for Grok
        """
        domain_context = TaxonomyPromptBuilder._get_domain_context(domain)
        specificity_instruction = TaxonomyPromptBuilder._get_specificity_instruction(specificity_level)

        # Add category limits to the prompt if provided
        limits_instruction = ""
        if category_limits:
            limits_instruction = f"\n**Category Limits per New Level:**\n"
            for i, limit in enumerate(category_limits, 1):
                limits_instruction += f"- New Level {i}: Maximum {limit} categories\n"
            limits_instruction += "\n"

        # Convert existing taxonomy to readable format
        existing_taxonomy_str = json.dumps(existing_taxonomy, indent=2)

        prompt = f"""You are an expert taxonomist specializing in {domain} knowledge organization. Your task is to extend an existing taxonomy by adding {f'{additional_depth} additional levels of ' if additional_depth else ''}subcategories based on the provided content.

**Domain Context:** {domain_context}

**Existing Taxonomy to Extend:**
{existing_taxonomy_str}

**Task Requirements:**
1. Analyze the existing taxonomy structure and identify where subcategories can be added
2. Add {additional_depth} additional levels of subcategories to the existing taxonomy
3. Ensure new categories are specific to the {domain} domain and consistent with existing categories
4. Use clear, descriptive names for new categories
5. Include confidence scores (0.0-1.0) for each new category based on content evidence
6. Make new categories mutually exclusive where possible
7. Focus on categories that would be useful for content classification and organization{specificity_instruction}{limits_instruction}

**Content to Analyze for Extension:**
{content}

**Output Format:**
Return a valid JSON object representing the extended taxonomy with this exact structure:
{{
  "name": "Root Category Name",
  "description": "Brief description of the root category",
  "children": [
    {{
      "name": "Level 1 Category",
      "description": "Description of this category",
      "confidence_threshold": 0.8,
      "children": [
        {{
          "name": "Level 2 Category",
          "description": "Description of subcategory",
          "confidence_threshold": 0.7,
          "children": [...]
        }}
      ]
    }}
  ]
}}

**Important:**
- Return ONLY the JSON object, no other text
- Ensure the JSON is valid and properly formatted
{f"- Add exactly {additional_depth} additional levels to the existing taxonomy" if additional_depth else "- Add appropriate subcategories to enhance the existing structure"}
- Include confidence_threshold for each new category (0.0-1.0)
- Make category names specific and actionable
- Preserve the existing taxonomy structure and add subcategories to appropriate nodes"""

        return prompt

    @staticmethod
    def _get_domain_context(domain: str) -> str:
        """
        Get domain-specific context for taxonomy generation.

        Args:
            domain: Knowledge domain

        Returns:
            Domain-specific guidance text
        """
        contexts = {
            "medical": "Focus on clinical conditions, symptoms, treatments, diagnostics, anatomy, and medical specialties. Consider diseases, medications, procedures, and patient care aspects.",
            "legal": "Focus on legal domains, case types, jurisdictions, legal procedures, contracts, regulations, and legal concepts. Consider civil law, criminal law, corporate law, etc.",
            "technical": "Focus on technical domains, software development, hardware, processes, methodologies, and technical concepts. Consider programming, systems, infrastructure, etc.",
            "business": "Focus on business domains, industries, departments, processes, strategies, and business concepts. Consider finance, marketing, operations, management, etc.",
            "scientific": "Focus on scientific domains, research areas, methodologies, theories, and scientific concepts. Consider physics, chemistry, biology, mathematics, etc.",
            "general": "Create a general-purpose taxonomy suitable for most content types. Focus on universal categories like people, places, events, concepts, etc."
        }

        return contexts.get(domain.lower(), contexts["general"])

    @staticmethod
    def _get_specificity_instruction(specificity_level: int) -> str:
        """
        Get specificity level instructions for taxonomy generation.

        Args:
            specificity_level: Level from 1-5, where higher numbers mean more specific categories

        Returns:
            Specificity instruction text for the prompt
        """
        specificity_instructions = {
            1: "**Specificity Level 1 (Most General):** Create very broad, high-level categories that capture the most general themes. Use overarching concepts that would apply to many different contexts. Avoid overly specific or niche categories.",
            2: "**Specificity Level 2 (1 level higher than Level 1):** Create moderately broad categories that are 1 specificity level higher than Level 1. Balance generality with some specificity while remaining more general than Level 3. Include common sub-themes but avoid highly specialized terms.",
            3: "**Specificity Level 3 (1 level higher than Level 2):** Create categories with a good balance of breadth and specificity that are 1 specificity level higher than Level 2. Include both general concepts and moderately specific subcategories as appropriate for the content, but remain more general than Level 4.",
            4: "**Specificity Level 4 (1 level higher than Level 3):** Create more detailed and specific categories that are 1 specificity level higher than Level 3. Focus on particular aspects, subtypes, and specific implementations rather than broad generalizations, but remain more general than Level 5.",
            5: "**Specificity Level 5 (1 level higher than Level 4):** Create very detailed, granular categories that are 1 specificity level higher than Level 4. Use highly specific terms, particular methodologies, and specialized concepts. Include fine distinctions and specific use cases."
        }

        instruction = specificity_instructions.get(specificity_level, specificity_instructions[3])
        return f"\n{instruction}\n\n"