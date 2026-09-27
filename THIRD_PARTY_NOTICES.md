# Third-party notices

This project's own code is MIT licensed (see `LICENSE`). It includes or bundles the following third-party material,
which remains under its own license:

- **`web/so101/`**: SO-101 arm URDF and STL meshes from [TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100)
  (Apache License 2.0). Used for the arm's inverse kinematics and the 3D viewer. See the upstream repository for the
  license text and attribution.
- **`web/viewer.bundle.js`, `web/diagnostics-viewer.bundle.js`**: built with esbuild from `web/viewer.js` /
  `web/diagnostics-viewer.js` and bundle [three.js](https://github.com/mrdoob/three.js) (MIT) and
  [urdf-loader](https://github.com/gkjohnson/urdf-loaders) (Apache License 2.0).

Not included, downloaded at setup time:

- [Qwen3-VL-4B-Instruct GGUF](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct-GGUF) (Apache License 2.0), run with
  [llama.cpp](https://github.com/ggml-org/llama.cpp) (MIT).
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) (AGPL-3.0), only used by the optional `--targeter yolo` mode.
- [rustypot](https://github.com/pollen-robotics/rustypot) (Apache License 2.0), the servo bus driver on the Raspberry Pi.
