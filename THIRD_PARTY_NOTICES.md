# Third-party notices

This repository contains project code and small browser/marker assets. Simulator
packages, robot USD files, the NVIDIA warehouse, model weights, and the LLM runtime
are obtained separately. Their licenses remain independent of any license for
this project's own code. This notice does not grant rights to external software
or assets and does not replace the applicable upstream license texts.

## Bundled browser assets: KaTeX

The local web interface includes the unmodified KaTeX **0.16.22** release's
JavaScript, CSS, and 60 font files under `src/opti_web/static/vendor/katex/`.

- Source: [KaTeX v0.16.22 release](https://github.com/KaTeX/KaTeX/releases/tag/v0.16.22).
- Release archive: `https://github.com/KaTeX/KaTeX/releases/download/v0.16.22/katex.tar.gz`.
- Archive SHA-256: `055cd79635251419ed908c334582a41dee341d8af05b4c5588277e145127a78f`.
- The JavaScript/CSS license is **MIT**. Copyright (c) 2013-2020 Khan Academy and
  other contributors. The complete upstream text is retained in
  [`src/opti_web/static/vendor/katex/LICENSE`](src/opti_web/static/vendor/katex/LICENSE),
  from the [versioned upstream license](https://github.com/KaTeX/KaTeX/blob/v0.16.22/LICENSE).
- The fonts' embedded license metadata specifies **SIL Open Font License 1.1**,
  with the copyright and reserved names reproduced below. Font licensing is
  separate from the MIT license for the browser code.
- `manifest.json` records each bundled file's hash; `src/opti_web/fetch_assets.py`
  can retrieve the official browser release again.

## External simulator and environment

### NVIDIA Isaac Sim 5.1.0

The setup installs `isaacsim[all,extscache]==5.1.0` from NVIDIA's package index.
Isaac Sim, Omniverse Kit, extension caches, and NVIDIA environment assets are not
included in this repository.

The official documentation distinguishes the Apache-2.0 licensed Isaac Sim
GitHub source from additional software and materials, including Kit, models, and
textures, governed by other terms. See the
[5.1.0 licensing overview](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/licenses-isaac-sim.html),
[NVIDIA Isaac Sim Additional Software and Materials License](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/license-isaac-sim-additional.html),
and [5.1.0 license index](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/common/legal.html).
The [Python installation instructions](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_python.html)
also require acceptance of the Omniverse license agreement. Setting an acceptance
environment variable is the user's acceptance of those terms, not a license
granted by this project.

At runtime the scene references
`Isaac/Environments/Simple_Warehouse/warehouse.usd` through the installed Isaac
asset resolver. The warehouse USD, meshes, textures, and downloaded asset cache
are not redistributed here. The asset-server content is not made immutable by
the Python package version; the repository does not claim a warehouse asset hash
pin. Applicable asset terms must be reviewed at their source.

### Isaac Lab

- Source: [isaac-sim/IsaacLab](https://github.com/isaac-sim/IsaacLab).
- Version: **v2.3.0**, commit `3c6e67bb5c7ada942a6d1884ab69338f57596f77`.
- Main license: **BSD-3-Clause**, Copyright (c) 2022-2025, The Isaac Lab Project
  Developers. See the [versioned LICENSE](https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/LICENSE).
- The upstream tree also contains a separate Apache-2.0
  [`LICENSE-mimic`](https://github.com/isaac-sim/IsaacLab/blob/3c6e67bb5c7ada942a6d1884ab69338f57596f77/LICENSE-mimic).
  File-specific terms take precedence over treating every upstream file as BSD.
- The checkout and packages are installed under `third_party/IsaacLab`; they are
  not bundled in this repository. The setup retains the upstream license files.

## External RB-Y1 robot and gripper

- Official source: [RainbowRobotics/rby1-sim-isaac](https://github.com/RainbowRobotics/rby1-sim-isaac).
- Fixed commit: `2417a2b2c83bc80b3ad605ab14f4d508d90089a9`.
- Referenced assets: `assets/model_v_1_2_a.usd`,
  `assets/gripper/rb_gripper/rb_gripper_left.usd`, and
  `assets/gripper/rb_gripper/rb_gripper_right.usd`.
- `configs/isaac/robot_assets.json` lists exact upstream file hashes;
  `scripts/isaac/fetch_robot.sh` retrieves and verifies a separate checkout.
  Vendor Python files, robot USD files, and gripper USD files are not included
  in this repository.

The [README at the fixed commit](https://github.com/RainbowRobotics/rby1-sim-isaac/blob/2417a2b2c83bc80b3ad605ab14f4d508d90089a9/README.md#license)
licenses source files bearing NVIDIA SPDX headers under **Apache-2.0**. Those
headers state Copyright (c) 2020-2025 NVIDIA CORPORATION & AFFILIATES.

No blanket license for the USD assets or the unheadered
`src/gripper_servers/` files was found in that commit. They must not be described
as Apache-2.0 licensed merely because neighboring source files have that header.
Obtaining them from the official repository does not itself resolve that missing
license scope; users must establish applicable permission with the upstream
provider before uses requiring it. This repository neither relicenses nor
redistributes those files.

`scripts/isaac/vendor_task.py` is a project adapter that loads the externally
obtained task. It removes an unused SDK bridge import in memory and leaves the
upstream checkout intact. The vendor SDK's closed binary wire codec and Docker
image are not shipped here. Project contact pads, tray, camera mount, racks, and
procedural worker geometry are authored by project code; they do not change the
license of the referenced robot or warehouse assets.

## External local LLM

### Qwen3-4B GGUF

- Publisher/repository: [Qwen/Qwen3-4B-GGUF](https://huggingface.co/Qwen/Qwen3-4B-GGUF).
- Fixed revision: `bc640142c66e1fdd12af0bd68f40445458f3869b`.
- File: `Qwen3-4B-Q4_K_M.gguf`.
- SHA-256: `7485fe6f11af29433bc51cab58009521f205840f5b4ae3a32fa7f92e8534fdf5`.
- License: **Apache-2.0**; see the
  [license at that revision](https://huggingface.co/Qwen/Qwen3-4B-GGUF/blob/bc640142c66e1fdd12af0bd68f40445458f3869b/LICENSE).

`scripts/local_llm_setup.py` downloads the weights, upstream license, and model
card into `third_party/llm_models/`. The weights are not redistributed here.

### llama.cpp

- Source: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp).
- Configured source tag: **v0.6.0**.
- The source associated with this notice resolved to commit
  `d81235049384534c167caea52b85a694f6103d14`.
- License: **MIT**, Copyright (c) 2023-2026 The ggml authors; see the
  [license at that commit](https://github.com/ggml-org/llama.cpp/blob/d81235049384534c167caea52b85a694f6103d14/LICENSE).

The downloader resolves the configured tag and records the actual commit and
archive hash in a local provenance file. It does not hardcode that resolved
commit as an immutable download constraint; retain the generated provenance when
building again. llama.cpp source, compiled executables, CUDA dependencies, and
model caches are not bundled in this repository.

## Other installed dependencies and generated assets

Python, PyTorch, CVXPY, OSQP, NumPy, SciPy, OpenCV, ROS 2 Humble, Nav2, and their
transitive dependencies are installed separately. Version constraints are in
`configs/isaac/` and the setup scripts. Each package retains its own upstream
license; this document is not a complete installed-environment license inventory
and does not assign one common license to those packages.

The small ArUco PNGs in `assets/isaac/` are generated from OpenCV's predefined
`DICT_4X4_50` dictionary, IDs 17, 18, and 19. The worker is procedural project
geometry, not a downloaded human character asset. No pretrained robot policy,
external character mesh, or NVIDIA environment texture is bundled here.

## KaTeX font copyright and license

The bundled TTF, WOFF, and WOFF2 metadata identifies:

```text
Copyright (c) 2009-2010, Design Science, Inc. (<www.mathjax.org>)
Copyright (c) 2014-2018 Khan Academy (<www.khanacademy.org>)
```

The Reserved Font Names are `KaTeX_AMS`, `KaTeX_Caligraphic`, `KaTeX_Fraktur`,
`KaTeX_Main`, `KaTeX_Math`, `KaTeX_SansSerif`, `KaTeX_Script`, `KaTeX_Size1`,
`KaTeX_Size2`, `KaTeX_Size3`, `KaTeX_Size4`, and `KaTeX_Typewriter`.
These font files remain under the following license. The official text is also
available from [SIL's Open Font License site](https://openfontlicense.org/open-font-license-official-text/).

```text
SIL OPEN FONT LICENSE Version 1.1 - 26 February 2007

PREAMBLE
The goals of the Open Font License (OFL) are to stimulate worldwide
development of collaborative font projects, to support the font creation
efforts of academic and linguistic communities, and to provide a free and
open framework in which fonts may be shared and improved in partnership
with others.

The OFL allows the licensed fonts to be used, studied, modified and
redistributed freely as long as they are not sold by themselves. The
fonts, including any derivative works, can be bundled, embedded,
redistributed and/or sold with any software provided that any reserved
names are not used by derivative works. The fonts and derivatives,
however, cannot be released under any other type of license. The
requirement for fonts to remain under this license does not apply
to any document created using the fonts or their derivatives.

DEFINITIONS
"Font Software" refers to the set of files released by the Copyright
Holder(s) under this license and clearly marked as such. This may
include source files, build scripts and documentation.

"Reserved Font Name" refers to any names specified as such after the
copyright statement(s).

"Original Version" refers to the collection of Font Software components as
distributed by the Copyright Holder(s).

"Modified Version" refers to any derivative made by adding to, deleting,
or substituting -- in part or in whole -- any of the components of the
Original Version, by changing formats or by porting the Font Software to a
new environment.

"Author" refers to any designer, engineer, programmer, technical
writer or other person who contributed to the Font Software.

PERMISSION & CONDITIONS
Permission is hereby granted, free of charge, to any person obtaining
a copy of the Font Software, to use, study, copy, merge, embed, modify,
redistribute, and sell modified and unmodified copies of the Font
Software, subject to the following conditions:

1) Neither the Font Software nor any of its individual components,
in Original or Modified Versions, may be sold by itself.

2) Original or Modified Versions of the Font Software may be bundled,
redistributed and/or sold with any software, provided that each copy
contains the above copyright notice and this license. These can be
included either as stand-alone text files, human-readable headers or
in the appropriate machine-readable metadata fields within text or
binary files as long as those fields can be easily viewed by the user.

3) No Modified Version of the Font Software may use the Reserved Font
Name(s) unless explicit written permission is granted by the corresponding
Copyright Holder. This restriction only applies to the primary font name as
presented to the users.

4) The name(s) of the Copyright Holder(s) or the Author(s) of the Font
Software shall not be used to promote, endorse or advertise any
Modified Version, except to acknowledge the contribution(s) of the
Copyright Holder(s) and the Author(s) or with their explicit written
permission.

5) The Font Software, modified or unmodified, in part or in whole,
must be distributed entirely under this license, and must not be
distributed under any other license. The requirement for fonts to
remain under this license does not apply to any document created
using the Font Software.

TERMINATION
This license becomes null and void if any of the above conditions are
not met.

DISCLAIMER
THE FONT SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO ANY WARRANTIES OF
MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT
OF COPYRIGHT, PATENT, TRADEMARK, OR OTHER RIGHT. IN NO EVENT SHALL THE
COPYRIGHT HOLDER BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
INCLUDING ANY GENERAL, SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL
DAMAGES, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
FROM, OUT OF THE USE OR INABILITY TO USE THE FONT SOFTWARE OR FROM
OTHER DEALINGS IN THE FONT SOFTWARE.
```
