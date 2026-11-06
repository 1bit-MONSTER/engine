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
# 1bit-engine.spec: the engine as an RPM for Fedora and openSUSE Tumbleweed
# (`dnf install ./1bit-engine-<version>.x86_64.rpm`, or zypper's `zypper install ./...`).
#
# Built by scripts/pack-distro.sh, which hands rpmbuild the payload tree as Source0, the files it
# carries as files.list, and the release as -Dver/-Drel:
#
#   PKG_NAME=1bit-engine PKG_VERSION=2026.45 scripts/pack-distro.sh rpm <payload> <out-dir>
#
# Nothing is compiled here: the payload is the engine, its HRX llama.cpp build, the sealed NPU
# kernel sets, the HSA runtime it dlopens and the shim that plugs a stock Lemonade into it, laid out
# at the paths it installs at (/opt/1bit/engine). A build tree is not an input, so there is no
# BuildRequires.
#
# No Requires either, and that is deliberate: the libraries this payload links (libhrx, libggml*,
# libhsa, libomp) ship inside it, so rpm's own ELF dependency generator reads only the sonames that
# are left - glibc, libstdc++, libgcc_s, libgomp, libcrypto/libssl, libzstd, libpcre2 - and resolves
# each to whatever the family calls that package. It also encodes the glibc floor the binary really
# has (libc.so.6(GLIBC_2.43)), so the package refuses to install on a distribution where it could not
# run instead of installing and failing to load.

%global debug_package %{nil}
# no brp-strip/brp-compress/find-debuginfo pass over the payload: it is shipped stripped, localized
# and byte-identical to what the .deb and the pacman package carry
%global __os_install_post %{nil}

Name:           1bit-engine
Version:        %{ver}
Release:        %{rel}
Summary:        1bit engine: HRX GPU kernels and the sealed NPU kernel sets, plug-in for stock Lemonade
License:        Apache-2.0 AND LicenseRef-1bit-Binary-License
URL:            https://1bit.gg/
BuildArch:      x86_64
Source0:        %{name}-%{version}.tar.gz

%description
The 1bit engine binary, its HRX llama.cpp build with the ggml-hrx backend and the HSA runtime library
it dlopens, the sealed NPU kernel sets, and a shim that makes a stock Lemonade (its llamacpp-hrx
recipe) run this engine with no patch and no fork.

Userspace only: no kernel modules, no Python at runtime.

%prep
%setup -q

%install
rm -rf %{buildroot}
mkdir -p %{buildroot}
cp -a ./. %{buildroot}/

%files -f %{_sourcedir}/files.list

%changelog
* %{chlogdate} bong-water-water-bong <bong-water-water-bong@1bit.gg> - %{ver}-%{rel}
- Package of the engine payload for this release (scripts/pack-distro.sh).
