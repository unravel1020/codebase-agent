# Sample repository

A tiny fixture repository used by `codebase-agent` tests and offline demos.

## Modules

- `app/calculator.py` - `Calculator` (add/subtract/multiply/divide) and `average`.
- `app/storage.py` - `JsonStorage` persists a list of records to a JSON file.
- `app/service.py` - `CalculatorService.compute()` calls the calculator and saves history.
- `app/utils.py` - `format_result` and `slugify` formatting helpers.

## Data flow

`CalculatorService.compute()` -> `Calculator` -> `JsonStorage.save()`, and
`CalculatorService.mean_result()` -> `average()`.
