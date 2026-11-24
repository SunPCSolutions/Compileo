"""Command-line interface for Compileo."""

# cython: annotation_typing=False

# Import with fallback for compiled/development compatibility
try:
    from compileo.cli_commands import cli
except ImportError:
    from src.compileo.cli_commands import cli

if __name__ == "__main__":
    cli()
