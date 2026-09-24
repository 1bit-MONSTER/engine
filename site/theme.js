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
*/

/* Theme modes: light (default) / auto / dark, from the 1bit.MONSTER site.
 *
 * Runs in <head> so data-theme is set on <html> before first paint (no flash).
 * The choice is kept under "1bit-theme"; no choice = light. Auto follows the OS
 * through the CSS media query in style.css; the buttons only need aria-pressed here.
 */
(function () {
  'use strict';
  var KEY = '1bit-theme';
  var MODES = ['light', 'auto', 'dark'];
  var root = document.documentElement;
  var store = null;
  try { store = window.localStorage; } catch (e) { /* private mode */ }

  function stored() {
    if (!store) return null;
    try { return store.getItem(KEY); } catch (e) { return null; }
  }
  function save(m) {
    if (!store) return;
    try { store.setItem(KEY, m); } catch (e) { /* ignore */ }
  }

  var mode = stored();
  if (MODES.indexOf(mode) < 0) mode = 'light';
  root.setAttribute('data-theme', mode);

  function wire() {
    var group = document.querySelector('.theme-switch');
    if (!group) return;
    var btns = group.querySelectorAll('.ts-btn');
    function mark(m) {
      for (var i = 0; i < btns.length; i++) {
        btns[i].setAttribute('aria-pressed', btns[i].getAttribute('data-choice') === m ? 'true' : 'false');
      }
    }
    for (var i = 0; i < btns.length; i++) {
      btns[i].addEventListener('click', function () {
        mode = this.getAttribute('data-choice');
        root.setAttribute('data-theme', mode);
        mark(mode);
        save(mode);
      });
    }
    mark(mode);
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', wire);
  else wire();
})();
