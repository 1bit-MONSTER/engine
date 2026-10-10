#!/usr/bin/env python3
# Copyright 2026 bong-water-water-bong
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "tools"))
import check_pins


class HrxValidationTest(unittest.TestCase):
    def test_changed_hrx_pin_requires_matching_validated_pin(self):
        error = check_pins.hrx_validation_error(
            check_pins.HRX_PATH, "a" * 40, "b" * 40
        )
        self.assertIn("long-prompt check", error)

    def test_validated_hrx_pin_is_allowed(self):
        self.assertIsNone(check_pins.hrx_validation_error(
            check_pins.HRX_PATH, "a" * 40, "a" * 40
        ))

    def test_other_submodule_does_not_require_hrx_validation(self):
        self.assertIsNone(check_pins.hrx_validation_error(
            "third_party/llama.cpp", "a" * 40, "b" * 40
        ))


if __name__ == "__main__":
    unittest.main()
