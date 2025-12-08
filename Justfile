default:
    @just --list

rs-test:
    @cargo t

py-test: py-dev
    @uv run --group tests pytest

py-dev:
    @maturin develop --uv --features py -m crates/rma-kinetics/Cargo.toml

rs-dev:
    @cargo check

docs-dev:
    @uv run --group docs mkdocs serve --watch docs

notebook name:
    @uv run marimo edit notebooks/{{name}}.py
