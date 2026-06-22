import runpy
from pathlib import Path

import gradio as gr
from PIL import Image

from python.voronoify_gui import PRESETS, PUBLIC_LIMITS, create_app, create_public_app
from python.voronoify_runner import BackendSpec


def test_presets_have_expected_values() -> None:
    assert PRESETS == {
        "Bold": (400, 0.8),
        "Balanced": (1200, 0.5),
        "Fine": (3000, 0.35),
    }


def test_create_app_smoke() -> None:
    app = create_app(
        [
            BackendSpec("fast", "Python fast (SciPy)", True),
            BackendSpec("native", "Native CUDA", True, executable=Path("/native")),
        ]
    )

    assert app.title == "Voronoify"
    backend_component = next(
        component for component in app.get_config_file()["components"] if component["type"] == "dropdown"
    )
    assert backend_component["props"]["choices"] == [
        ("Python fast (SciPy)", "fast"),
        ("Native CUDA", "native"),
    ]


def test_public_app_is_fast_only_without_backend_selector() -> None:
    app = create_public_app(
        [
            BackendSpec("fast", "Python fast (SciPy)", True),
            BackendSpec("native", "Native CUDA", True, executable=Path("/native")),
        ]
    )
    config = app.get_config_file()
    component_types = [component["type"] for component in config["components"]]
    input_image = next(
        component
        for component in config["components"]
        if component["type"] == "image" and component["props"].get("label") == "Input image"
    )

    assert "dropdown" not in component_types
    assert "state" in component_types
    assert input_image["props"]["value"]["orig_name"] == "demo_wave.jpg"
    assert PUBLIC_LIMITS.max_source_pixels == 25_000_000
    assert PUBLIC_LIMITS.max_output_pixels == 2_000_000

    demo_path = Path(__file__).resolve().parents[2] / "img" / "demo_wave.jpg"
    with Image.open(demo_path) as demo_image:
        assert demo_image.size == (1280, 720)
    assert demo_path.stat().st_size < 10 * 1024 * 1024


def test_hosted_entry_point_does_not_launch_when_imported(monkeypatch) -> None:
    def fail_launch(*_args, **_kwargs):
        raise AssertionError("The hosted app launched during import")

    monkeypatch.setattr(gr.Blocks, "launch", fail_launch)
    app_path = Path(__file__).resolve().parents[2] / "app.py"
    namespace = runpy.run_path(str(app_path), run_name="hosted_app_test")

    assert namespace["demo"]._queue.max_size == 8
