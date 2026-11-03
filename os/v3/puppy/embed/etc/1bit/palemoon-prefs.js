/*
  Copyright 2026 bong-water-water-bong
  SPDX-License-Identifier: Apache-2.0

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.

  Pale Moon's first run on 1bit OS (cycle 17 item 9).

  This file lives in /etc/1bit and is COPIED into <app>/defaults/pref/1bit-os.js by 1bit-lemonade-webapp just
  before the browser starts. Why not shipped straight into the application directory: woof-CE installs Pale Moon
  by moving the whole unpacked tree onto /opt/palemoon (`mv palemoon/ /opt/palemoon`), so a directory that exists
  there at that moment breaks the build ("cannot move ... Directory not empty", measured 2026-11-03). Every Gecko
  browser reads <app>/defaults/pref/*.js before it has a profile - channel-prefs.js is loaded the same way - so
  the copy lands before the first profile exists, and no per-profile path has to be guessed.

  The only question the appliance answers here is the first-run "make Pale Moon the default browser?" prompt: on
  a demo box the browser is the point, and a modal the owner has to dismiss on camera is not. The window size is
  not a pref - 1bit-lemonade-webapp maximises the window it started with xdotool.
*/
pref("browser.shell.checkDefaultBrowser", false);
