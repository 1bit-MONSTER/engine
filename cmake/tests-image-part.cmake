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
#
# engine#2324: the one place that knows what a picture arrives as -- a data: URL -- turning it into the raw RGB plus geometry
# that npu/vision_image.h and npu/vision_tower.h already take. Header-only, no model, no device.
add_executable(npu_image_part_test tests/npu_image_part_test.cpp)
target_include_directories(npu_image_part_test PRIVATE app)
add_test(NAME npu_image_part COMMAND npu_image_part_test)
add_dependencies(onebit npu_image_part_test)
