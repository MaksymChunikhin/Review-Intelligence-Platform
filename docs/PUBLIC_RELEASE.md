# Public Repository Release Checklist

## Included

- application and ML/NLP source code;
- automated tests and schemas;
- versioned configuration;
- notebooks with outputs removed;
- architecture and methodology documentation;
- aggregate, text-free portfolio metrics.

## Excluded

- `.env` and all API credentials;
- raw and processed Amazon datasets;
- fitted model binaries and retrieval indexes;
- complete review text and user identifiers;
- generated workbooks, review queues, and historical report archives.

## Owner actions before the first public push

1. License decision: no open-source license. The public repository is provided
   for portfolio and code-review purposes under default copyright restrictions.
2. Add two or three dashboard screenshots under `docs/images/` and reference
   them near the top of `README.md`.
3. Create an empty GitHub repository without an auto-generated README,
   `.gitignore`, or license.
4. Review `git status --short` and the exact staged set before committing.
5. Enable GitHub secret scanning and push protection.
6. Push only after the local secret, path, and file-size checks pass.

Do not use `git add -f` to bypass the repository exclusions.
