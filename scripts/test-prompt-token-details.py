#!/usr/bin/env python3
"""CPU-only regression for the Compose serve argv shared by both ranks."""
import shlex
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class PromptTokenDetailsTests(unittest.TestCase):
    def test_compose_serve_args(self):
        source = (ROOT / "docker-compose.dspark.yml").read_text()
        marker = "exec /opt/env/bin/vllm serve "
        self.assertEqual(source.count(marker), 1)
        argv = shlex.split(source.split(marker, 1)[1], comments=True)
        self.assertEqual(argv.count("--enable-prompt-tokens-details"), 1)
        self.assertIn("--enable-prefix-caching", argv)
        self.assertNotIn("--enable-prompt-token-details", argv)
        self.assertNotIn("--no-enable-prompt-tokens-details", argv)
        self.assertIn("${NODE_RANK}", argv)
        self.assertIn("${HEADLESS:+--headless}", argv)


if __name__ == "__main__":
    unittest.main()
