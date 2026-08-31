import json
import tempfile
import unittest
from pathlib import Path

from solar_filament.cli import main


class CliTests(unittest.TestCase):
    def test_audit_writes_machine_readable_report(self):
        data_root = Path("filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
        if not data_root.exists():
            self.skipTest("competition data is not attached")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "audit.json"

            exit_code = main(["audit", str(data_root), "--output", str(output)])
            report = json.loads(output.read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(report["physical_images"], 707)

    def test_folds_command_writes_every_physical_image_once(self):
        data_root = Path("filament-segmentation-2026/MAGFiLO_1.0_Kaggle_2026")
        if not data_root.exists():
            self.skipTest("competition data is not attached")
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "folds.json"

            exit_code = main(
                ["folds", str(data_root), "--output", str(output), "--n-folds", "5"]
            )
            folds = json.loads(output.read_text())

        self.assertEqual(exit_code, 0)
        self.assertEqual(len(folds), 707)
        self.assertEqual(set(folds.values()), {0, 1, 2, 3, 4})


if __name__ == "__main__":
    unittest.main()
