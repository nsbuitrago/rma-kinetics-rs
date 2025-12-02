default:
    @just --list

rs-test:
    @cargo t

py-test: py-dev
    @uv run pytest

py-dev:
    @maturin develop --uv --features py -m crates/rma-kinetics/Cargo.toml

rs-dev:
    @cargo check

docs-dev:
    @uv run mkdocs serve --watch docs
