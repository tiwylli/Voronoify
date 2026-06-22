#!/usr/bin/env python3
"""Local Gradio web interface for Voronoify."""

from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from python.voronoify_runner import (  # noqa: E402
    BackendSpec,
    JobCancelled,
    RunConfig,
    RunnerError,
    cancel_job,
    detect_backends,
    run_job,
)


PRESETS = {
    "Bold": (400, 0.8),
    "Balanced": (1200, 0.5),
    "Fine": (3000, 0.35),
}


def _backend_status(backends: list[BackendSpec]) -> str:
    lines = []
    for backend in backends:
        if backend.available:
            lines.append(f"- **{backend.label}:** available")
        else:
            lines.append(f"- **{backend.label}:** unavailable — {backend.unavailable_reason}")
    return "\n".join(lines)


def _coerce_integer(value: int | float | None, name: str) -> int:
    if value is None or isinstance(value, bool) or int(value) != value:
        raise RunnerError(f"{name} must be an integer.")
    return int(value)


def create_app(backends: list[BackendSpec] | None = None) -> gr.Blocks:
    backend_specs = detect_backends() if backends is None else backends
    backend_map = {backend.identifier: backend for backend in backend_specs}
    choices = [(backend.label, backend.identifier) for backend in backend_specs if backend.available]
    fast_backend = backend_map.get("fast")
    default_backend = "fast" if fast_backend and fast_backend.available else None
    if default_backend is None and choices:
        default_backend = choices[0][1]

    def generate(image_path, backend_id, cells, jitter, seed, request: gr.Request):
        yield gr.skip(), "Running…"
        try:
            config = RunConfig(
                image_path=image_path or "",
                backend=backend_id or "",
                cells=_coerce_integer(cells, "Cells"),
                jitter=float(jitter),
                seed=_coerce_integer(seed, "Seed"),
            )
            result = run_job(config, request.session_hash or "local", backends=backend_specs)
            yield result.image, f"Completed with {result.backend.label} in {result.elapsed_seconds:.2f}s."
        except JobCancelled:
            yield gr.skip(), "Generation cancelled."
        except (RunnerError, TypeError, ValueError) as exc:
            yield gr.skip(), f"Error: {exc}"

    def cancel(request: gr.Request):
        was_running = cancel_job(request.session_hash or "local")
        return "Generation cancelled." if was_running else "No generation is currently running."

    def cleanup(request: gr.Request):
        cancel_job(request.session_hash or "local")

    with gr.Blocks(
        title="Voronoify",
        analytics_enabled=False,
        delete_cache=(3600, 3600),
        fill_width=True,
    ) as app:
        gr.Markdown(
            "# Voronoify\n"
            "Turn an image into a Voronoi mosaic using the processing backend available on this machine."
        )
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=320):
                input_image = gr.Image(
                    label="Input image",
                    type="filepath",
                    sources=["upload"],
                    height=430,
                )
                backend = gr.Dropdown(
                    choices=choices,
                    value=default_backend,
                    label="Backend",
                    info="Only runnable backends can be selected.",
                    interactive=bool(choices),
                )
                with gr.Accordion("Backend availability", open=False):
                    gr.Markdown(_backend_status(backend_specs))
            with gr.Column(scale=1, min_width=320):
                output_image = gr.Image(
                    label="Voronoi output",
                    format="png",
                    height=430,
                    interactive=False,
                    buttons=["download", "fullscreen"],
                )

        gr.Markdown("### Style")
        with gr.Row():
            preset_buttons = {name: gr.Button(name, variant="secondary") for name in PRESETS}
        with gr.Row():
            cells = gr.Slider(1, 5000, value=PRESETS["Balanced"][0], step=1, label="Cells")
            jitter = gr.Slider(0, 1, value=PRESETS["Balanced"][1], step=0.05, label="Jitter")
            seed = gr.Number(value=0, minimum=0, precision=0, label="Seed")

        with gr.Row():
            generate_button = gr.Button("Generate", variant="primary", interactive=bool(choices))
            cancel_button = gr.Button("Cancel", variant="stop")
        status = gr.Markdown("Ready.")

        for name, button in preset_buttons.items():
            button.click(
                fn=lambda preset=name: PRESETS[preset],
                inputs=None,
                outputs=[cells, jitter],
                queue=False,
                api_visibility="private",
            )

        generate_event = generate_button.click(
            fn=generate,
            inputs=[input_image, backend, cells, jitter, seed],
            outputs=[output_image, status],
            concurrency_limit=1,
            concurrency_id="voronoify-processing",
            trigger_mode="once",
            api_visibility="private",
        )
        cancel_button.click(
            fn=cancel,
            inputs=None,
            outputs=status,
            queue=False,
            cancels=[generate_event],
            api_visibility="private",
        )
        app.unload(cleanup)

    return app


def main() -> None:
    app = create_app()
    app.queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1",
        share=False,
        inbrowser=True,
        show_error=False,
    )


if __name__ == "__main__":
    main()
