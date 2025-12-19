# Branch-Specific Notes

## Client Folders (jpeg-byte branch only)

The `client/python_client` and `client/cpp_client` folders are **only maintained in the `jpeg-byte` branch**.

### For main/master branch:

To exclude these folders from the main branch, update `.gitignore`:

```bash
# In main/master branch, uncomment these lines in .gitignore:
client/python_client/
client/cpp_client/
```

### For jpeg-byte branch:

Keep these lines commented out in `.gitignore` to track the client folders:

```bash
# client/python_client/
# client/cpp_client/
```

### Switching branches:

When switching between branches, Git will handle the folder visibility automatically based on the branch's `.gitignore` configuration.

