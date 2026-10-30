// Copyright 2026 bong-water-water-bong
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
// Unless required by applicable law or agreed to in writing, software
// limitations under the License.

// ort_info_main.cpp — standalone entry point for the probe. The engine's single
// binary (`1bit ort-info`) reaches onebit_ort_info_main() through app/main.cpp
// instead, so this file is only built for the standalone/ctest target.

int onebit_ort_info_main(int argc, char** argv);

int main(int argc, char** argv) { return onebit_ort_info_main(argc, argv); }
