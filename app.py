#!/usr/bin/env python3
"""Hosted Hugging Face Spaces entry point for the public Voronoify demo."""

from python.voronoify_gui import create_public_app


demo = create_public_app()
demo.queue(max_size=8, default_concurrency_limit=1)


if __name__ == "__main__":
    demo.launch(
        share=False,
        show_error=False,
        max_file_size="10mb",
        enable_monitoring=False,
    )
