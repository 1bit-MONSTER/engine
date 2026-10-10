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

sys.path.insert(0, os.path.dirname(__file__))
import hrx_long_prompt


class LongPromptTest(unittest.TestCase):
    def test_needle_is_on_expected_record_and_question_is_last(self):
        prompt = hrx_long_prompt.build_prompt()
        records = prompt.split("\n\n", 1)[1].split("\n\n", 1)[0].splitlines()
        self.assertEqual(len(records), 320)
        self.assertIn("secret vault code is 7341", records[202])
        self.assertTrue(prompt.endswith(
            "What is the secret vault code? Answer with the number only."
        ))


if __name__ == "__main__":
    unittest.main()
