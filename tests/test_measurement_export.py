import json

import numpy as np

from dendro.imaging.path_sampler import widths_to_csv
from dendro.imaging.viewer import MeasurementSession
from dendro.reference.tucson_parser import load_measurement_session


def test_widths_to_csv_exports_canonical_oldest_to_newest(tmp_path):
    output = tmp_path / "measurements.csv"
    csv_text = widths_to_csv(
        np.array([0.2, 0.5, 0.9]),
        orientation="bark_to_pith",
        output_path=output,
    )

    lines = csv_text.splitlines()
    assert lines[0] == "ring_index,width_mm"
    assert lines[1] == "1,0.900"
    assert lines[-1] == "3,0.200"
    assert output.exists()


def test_measurement_session_round_trips_exported_widths(tmp_path):
    session = MeasurementSession(
        image_path="sample.tiff",
        dpi=1200,
        image_shape=(100, 200),
        ring_widths_mm_bark_to_pith=[0.2, 0.5, 0.9],
        exported_widths_mm_oldest_to_newest=[0.9, 0.5, 0.2],
        warnings=["Low profile contrast"],
        is_finalized=True,
    )

    path = tmp_path / "session.json"
    session.save(path)

    payload = json.loads(path.read_text())
    assert payload["export_orientation"] == "oldest_to_newest"

    df = load_measurement_session(path)
    assert list(df["width"]) == [0.9, 0.5, 0.2]
