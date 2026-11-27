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

  1bit OS update page (update.html): served from its own origin, 127.0.0.1:8081, whose CSP is default-src 'self'
  - so no inline script or style.
*/
function $(id) { return document.getElementById(id); }
function show(id, text) { $(id).textContent = text; }
function post(action, confirm) {
  show('why', '');
  fetch('/1bit/update/' + action, { method: 'POST', headers: { 'Content-Type': 'application/json', 'X-1bit-Update': '1' },
                                    body: JSON.stringify({ confirm: confirm }) })
    .then(function (r) { return r.json().then(function (j) { return [r.status, j]; }); })
    .then(function (a) { if (a[0] !== 202) show('why', a[1].error || ('HTTP ' + a[0])); poll(); })
    .catch(function (e) { show('why', 'the proxy did not answer: ' + e); });
}
function poll() {
  fetch('/1bit/update/progress').then(function (r) { return r.json(); }).then(function (j) {
    var job = j.job || {};
    show('log', (job.action ? job.action + (job.running ? ' - running' : ' - finished, exit ' + job.rc) + '\n\n' : '') + (j.log || ''));
    if (job.running) setTimeout(poll, 2000);
  }).catch(function () {});
}
function load() {
  fetch('/1bit/update/status').then(function (r) { return r.json(); }).then(function (j) {
    show('status', j.status); show('check', j.check);
    $('install').hidden = !j.offered; show('offer', j.offered || '');
    $('roll').hidden = !j.rollback_to; show('slot', j.rollback_to || '');
    if (!j.admin) show('why', 'Install, roll back and reboot need root on this box (its own desktop session, or ' +
                                'curl from a root shell); this browser is not running as root.');
    poll();
  }).catch(function (e) { show('status', 'the proxy did not answer: ' + e); });
}
$('bi').onclick = function () { post('latest', $('ci').value.trim()); };
$('br').onclick = function () { post('rollback', $('cr').value.trim()); };
$('bb').onclick = function () { post('reboot', $('cb').value.trim()); };
load();
