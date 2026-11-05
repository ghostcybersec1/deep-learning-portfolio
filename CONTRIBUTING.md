# Portfolio project checklist

New projects belong in `projects/<descriptive-project-name>/`. Keep each project
self-contained and use a lowercase, hyphenated folder name.

## Required project contents

- `README.md` with the problem, approach, setup, results, limitations, and
  responsible-use notes
- reproducible dependency metadata
- reusable source code separated from exploratory notebooks
- automated tests that run without private datasets
- an explicit data contract and instructions for obtaining authorized inputs
- a project-specific `.gitignore`

## Quality gate

Before committing a project:

1. Run its complete test suite.
2. Validate notebooks and clear outputs containing local paths or identities.
3. Compare the model with an appropriate simple baseline.
4. Document train, validation, and test separation to prevent leakage.
5. Verify that generated artifacts and large datasets are ignored.
6. Scan the exact Git candidate set for credentials and personal information.
7. Confirm that every included dataset or asset may legally be redistributed.

## Privacy and security

Never commit `.env` files, API keys, cloud credentials, access tokens, private
keys, personal contact details, local home-directory paths, raw restricted data,
or trained checkpoints containing sensitive metadata. Use synthetic fixtures in
tests and environment variables for local configuration.
