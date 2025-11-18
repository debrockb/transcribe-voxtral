---
description: Bump the version number for a new release
---

Help the user bump the version number and prepare a new release.

1. Check the current version in VERSION file
2. Ask the user what type of version bump (major, minor, patch)
3. Calculate the new version number
4. Update version in these files:
   - VERSION
   - config.json (app.version)
5. Run tests to ensure everything works
6. Show the user a git commit command template:
```bash
git add VERSION config.json
git commit -m "Release v{NEW_VERSION} - {DESCRIPTION}

{CHANGELOG}

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>"
git tag v{NEW_VERSION}
git push && git push --tags
```

7. Ask if they want to create a GitHub release
