---
description: Run security-focused tests and checks
---

Run security-specific tests to verify path traversal protection and other security measures:

1. Run the path traversal security tests:
```bash
test_venv/bin/pytest tests/test_api.py::TestPathTraversalSecurity -v
```

2. Check for common security issues in the codebase:
   - Review endpoints for authentication requirements
   - Verify input validation in all API endpoints
   - Check for SQL injection vulnerabilities (if using SQL)
   - Review file upload restrictions
   - Check for XSS vulnerabilities in responses

3. Provide a security audit summary with:
   - Results of security tests
   - Any potential vulnerabilities found
   - Recommendations for improvements
