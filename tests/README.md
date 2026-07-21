# Static Test Coverage

Recommended pre-release checks:

```bash
PYTHONDONTWRITEBYTECODE=1 python tests/test_static_contract.py
```

The test script covers:

- CLI help;
- fixture `validate-output`;
- static source check that `run-single` has an authorization guard before importing the real Huatuo runtime bridge;
- absence of stale model/layer labels in HTML;
- absence of local cache paths and other nonportable strings in the staging tree.
