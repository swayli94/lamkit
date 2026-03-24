# lamkit

A toolkit for stress analysis and failure prediction of composite laminates with holes and joints.

## Development setup

```bash
pip install -e .[dev,docs]
pytest
```

## Build package

```bash
python -m build
```

## Build docs

```bash
cd docs
sphinx-build -b html . _build/html
```
