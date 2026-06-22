from python.voronoify_gui import PRESETS, create_app
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
            BackendSpec("native", "Native CUDA", False, "Not built."),
        ]
    )

    assert app is not None
    assert app.title == "Voronoify"
