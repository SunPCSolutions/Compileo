"""
Plugin GUI page hosting utilities.

Provides a thin extension layer so plugins can contribute Streamlit pages
without breaking core GUI behavior when no plugins are installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional
import logging

from src.compileo.features.plugin.manager import plugin_manager

logger = logging.getLogger(__name__)


@dataclass
class PluginPageDefinition:
    """Normalized plugin page definition used by the GUI host."""

    key: str
    label: str
    section: str
    order: int
    renderer: Callable[..., Any]


def _parse_order(raw_order: Any) -> int:
    """Parse page order with safe fallback."""
    try:
        return int(raw_order)
    except Exception:
        return 1000


def _normalize_page_value(page_key: str, value: Any) -> Optional[PluginPageDefinition]:
    """Normalize plugin extension values to PluginPageDefinition."""
    if isinstance(value, tuple) and len(value) == 2:
        value = value[0]

    # Class-based extension support: class may expose renderer() and optional metadata.
    if isinstance(value, type):
        render_attr = getattr(value, "renderer", None)
        if callable(render_attr):
            return PluginPageDefinition(
                key=str(getattr(value, "key", page_key)),
                label=str(getattr(value, "label", page_key)),
                section=str(getattr(value, "section", "plugin")),
                order=_parse_order(getattr(value, "order", 1000)),
                renderer=render_attr,
            )

    if callable(value):
        return PluginPageDefinition(
            key=page_key,
            label=page_key,
            section="plugin",
            order=1000,
            renderer=value,
        )

    if isinstance(value, dict):
        render_fn = value.get("renderer")
        if not callable(render_fn):
            return None

        return PluginPageDefinition(
            key=str(value.get("key") or page_key),
            label=str(value.get("label") or page_key),
            section=str(value.get("section") or "plugin"),
            order=_parse_order(value.get("order")),
            renderer=render_fn,
        )

    return None


def get_plugin_pages_for_user(owner_id: Optional[str] = None) -> List[PluginPageDefinition]:
    """
    Retrieve plugin page definitions for a user.

    Returns an empty list on any failure to preserve core GUI behavior.
    """
    try:
        raw_pages: Dict[str, Any] = plugin_manager.get_extensions("compileo.gui.page", owner_id=owner_id)
    except Exception as exc:
        logger.debug("Failed to load plugin GUI pages: %s", exc)
        return []

    pages: List[PluginPageDefinition] = []
    seen_keys = set()

    for page_key, raw_value in raw_pages.items():
        try:
            page_def = _normalize_page_value(page_key, raw_value)
            if not page_def:
                continue
            if page_def.key in seen_keys:
                continue
            seen_keys.add(page_def.key)
            pages.append(page_def)
        except Exception as exc:
            logger.debug("Skipping invalid plugin page '%s': %s", page_key, exc)

    pages.sort(key=lambda p: (p.section.lower(), p.order, p.label.lower()))
    return pages


def render_plugin_page(page_key: str, owner_id: Optional[str] = None) -> bool:
    """
    Render a plugin page by key.

    Returns:
        True if a plugin page was found and rendered, otherwise False.
    """
    pages = get_plugin_pages_for_user(owner_id=owner_id)
    page = next((p for p in pages if p.key == page_key), None)
    if not page:
        return False

    try:
        page.renderer()
        return True
    except Exception:
        logger.exception("Plugin page renderer failed for key '%s'", page_key)
        raise
