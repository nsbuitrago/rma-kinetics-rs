{
  description = "A Nix-flake-based development environment";

  inputs = {
    nixpkgs.url = "https://flakehub.com/f/NixOS/nixpkgs/0.1"; # unstable Nixpkgs
    fenix = {
      url = "https://flakehub.com/f/nix-community/fenix/0.1";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    { self, ... }@inputs:

    let
      supportedSystems = [
        "x86_64-linux"
        "aarch64-linux"
        "x86_64-darwin"
        "aarch64-darwin"
      ];
      forEachSupportedSystem =
        f:
        inputs.nixpkgs.lib.genAttrs supportedSystems (
          system:
          f {
            pkgs = import inputs.nixpkgs {
              inherit system;
              overlays = [
                inputs.self.overlays.default
              ];
            };
          }
        );
    in
    {
      overlays.default = final: prev: {
        rustToolchain =
          with inputs.fenix.packages.${prev.stdenv.hostPlatform.system};
          combine (
            with stable;
            [
              clippy
              rustc
              cargo
              rustfmt
              rust-src
            ]
          );
      };

      devShells = forEachSupportedSystem (
        { pkgs }:
        {
          default = pkgs.mkShell {
            packages =
              (with pkgs; [
                rustToolchain
                openssl
                pkg-config
                cargo-deny
                cargo-edit
                cargo-watch
                rust-analyzer
                uv
                maturin
                just
                git
              ])
              ++ pkgs.lib.optionals pkgs.stdenv.isLinux [
                pkgs.python313
                pkgs.zlib
                pkgs.stdenv.cc.cc.lib
              ];

            env =
              {
                # Required by rust-analyzer
                RUST_SRC_PATH = "${pkgs.rustToolchain}/lib/rustlib/src/rust/library";
              }
              // pkgs.lib.optionalAttrs pkgs.stdenv.isLinux {
                UV_PYTHON_DOWNLOADS = "never";
                UV_PYTHON_PREFERENCE = "only-system";
                UV_PYTHON = "${pkgs.python313}/bin/python3.13";
                LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
                  pkgs.stdenv.cc.cc.lib
                  pkgs.zlib
                ];
              };
          };
        }
      );
    };
}
