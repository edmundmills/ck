{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  # Build deps for ck with the local-ONNX providers. `ort` uses
  # download-binaries (static onnxruntime) + tls-native (openssl), so the
  # toolchain needs openssl + pkg-config. The resulting binary statically
  # links onnxruntime — no libonnxruntime.so / ORT_DYLIB_PATH at runtime.
  buildInputs = [
    pkgs.cargo
    pkgs.rustc
    pkgs.pkg-config
    pkgs.openssl
    pkgs.stdenv.cc.cc.lib
  ];
  RUST_BACKTRACE = "1";
}
