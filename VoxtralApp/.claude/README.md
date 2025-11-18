# Claude Code Commands for Voxtral

This directory contains custom slash commands for Claude Code to streamline development of the Voxtral transcription application.

## Available Commands

### `/run-tests`
Runs the complete test suite using pytest, excluding model-dependent tests. Provides detailed results including coverage information.

### `/check-security`
Runs security-focused tests, specifically the path traversal security tests and provides a security audit of the codebase.

### `/start-app`
Starts the Flask application server, checking for existing instances and providing helpful information about the running app.

### `/check-style`
Runs code quality checks using flake8, black, and isort. Reports any style violations and formatting issues, with options to auto-fix.

### `/bump-version`
Assists with version bumping by updating VERSION and config.json files, running tests, and providing git commands for tagging and release.

### `/debug-app`
Comprehensive debugging assistant that checks system status, dependencies, Flask app status, model configuration, and helps diagnose common issues.

## Usage

To use these commands in Claude Code, simply type the slash command in your conversation:

```
/run-tests
```

Claude Code will automatically load the command instructions and execute the appropriate actions.

## Adding New Commands

To add a new command:

1. Create a new `.md` file in `.claude/commands/`
2. Add frontmatter with a description:
   ```markdown
   ---
   description: Your command description
   ---
   ```
3. Write the instructions for Claude Code to follow
4. Save and the command will be automatically available

## Command Structure

Each command file should:
- Have a descriptive filename (becomes the command name)
- Include frontmatter with a description
- Provide clear, actionable instructions
- Include example bash commands when appropriate
- Specify what output/summary should be provided to the user

## Learn More

For more information about Claude Code commands, visit:
https://code.claude.com/docs
