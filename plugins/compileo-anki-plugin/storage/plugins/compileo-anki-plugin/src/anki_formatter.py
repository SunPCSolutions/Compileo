import json
from typing import List, Dict, Any, Union

class AnkiOutputFormatter:
    """
    Formats datasets into Anki-compatible text files.
    Format: question;answer;tags

    This plugin receives the standard JSON output from Compileo and converts it
    to Anki-compatible semicolon-separated text format.
    """

    def format(self, dataset_content: Union[str, List[Dict[str, Any]]]) -> str:
        """
        Formats the dataset content into Anki-compatible semicolon-separated text.

        Args:
            dataset_content: Either a JSON string (standard Compileo output) or
                           a list of dictionaries (raw data)

        Returns:
            Anki-compatible text format: question;answer;tags
        """
        # Handle JSON string input (standard Compileo output)
        if isinstance(dataset_content, str):
            try:
                # Parse the JSON string to get the list of items
                dataset_content = json.loads(dataset_content)
                print(f"📄 Parsed JSON input, found {len(dataset_content)} items")
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON input for Anki formatter: {e}")

        # Ensure we have a list of dictionaries
        if not isinstance(dataset_content, list):
            raise ValueError("Anki formatter requires a list of items (or JSON string containing a list)")

        print(f"🔄 Converting {len(dataset_content)} items to Anki format")

        lines = []

        for item in dataset_content:
            if isinstance(item, dict):
                # Standard dictionary format
                front = item.get("question", "")
                back = item.get("answer", "")

                # Optional: Add reasoning to the back if present
                reasoning = item.get("reasoning", "")
                if reasoning and back:
                    back = f"{back}\n\nReasoning: {reasoning}"
                elif reasoning:
                    back = f"Reasoning: {reasoning}"

                # Tags: Could be derived from categories or other metadata
                tags = item.get("tags", "")
                if isinstance(tags, list):
                    tags = " ".join(tags)

            elif isinstance(item, str):
                # Handle string input - check if it's a long concatenated string
                if len(item) > 1000 and item.count(";") > 5:  # Long string with multiple Q&A pairs
                    print(f"📝 Detected long concatenated string ({len(item)} chars), splitting into individual Q&A pairs")
                    # Split by pattern: look for question endings followed by new questions
                    # Pattern: split before "A [digit]:" or similar patterns that indicate new questions
                    import re

                    # Split on patterns like "A 28-year-old", "A 65-year-old", etc. (new question starts)
                    qa_pairs = re.split(r'(?=A \d+-year-old|A \d+-year|A \d+)', item)

                    # Clean up the pairs
                    qa_pairs = [pair.strip() for pair in qa_pairs if pair.strip()]

                    print(f"✂️ Split into {len(qa_pairs)} Q&A pairs")

                    # Process each pair
                    for pair in qa_pairs:
                        if ";" in pair:
                            parts = pair.split(";", 1)
                            if len(parts) == 2:
                                front = parts[0].strip()
                                back = parts[1].strip()

                                # Clean up HTML line breaks
                                back = back.replace("<br>", "\n")

                                # Format for Anki
                                front_formatted = self._format_field(front)
                                back_formatted = self._format_field(back)
                                tags_formatted = ""

                                line = f"{front_formatted};{back_formatted}"
                                lines.append(line)
                    continue  # Skip the rest of the loop since we processed all pairs
                else:
                    # Single Q&A pair string
                    if ";" in item:
                        parts = item.split(";", 1)
                        if len(parts) == 2:
                            front = parts[0].strip()
                            back = parts[1].strip()

                            # Clean up HTML line breaks
                            back = back.replace("<br>", "\n")

                            tags = ""
                        else:
                            front = item
                            back = ""
                            tags = ""
                    else:
                        front = item
                        back = ""
                        tags = ""
            else:
                print(f"⚠️ Skipping invalid item type: {type(item)}")
                continue

            # Format fields for Anki (escape semicolons and quotes)
            front_formatted = self._format_field(str(front))
            back_formatted = self._format_field(str(back))
            tags_formatted = self._format_field(str(tags))

            # Construct Anki line: question;answer;tags
            if tags_formatted:
                line = f"{front_formatted};{back_formatted};{tags_formatted}"
            else:
                line = f"{front_formatted};{back_formatted}"

            lines.append(line)

        result = "\n".join(lines)
        print(f"✅ Generated Anki format with {len(lines)} cards")
        return result

    def _format_field(self, text: str) -> str:
        """
        Formats a single field:
        1. Replaces newlines with <br> (HTML mode)
        2. Escapes double quotes by doubling them
        3. Wraps in quotes if it contains separator (;) or quotes
        """
        # Replace newlines with <br> for HTML support
        # Note: User instructions said: "Escaped multi-lines will not work correctly... please use HTML newlines instead"
        text = text.replace("\n", "<br>")
        
        # Check if we need to escape (if contains semicolon or quotes)
        if ";" in text or '"' in text:
            # Escape double quotes by doubling them
            text = text.replace('"', '""')
            # Wrap in quotes
            return f'"{text}"'
        
        return text