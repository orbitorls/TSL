import re

with open("pyproject.toml", "r") as f:
    content = f.read()

# Replace top level ruff configs to tool.ruff.lint
content = content.replace("select = [", "[tool.ruff.lint]\nselect = [")
content = content.replace("[tool.ruff.per-file-ignores]", "[tool.ruff.lint.per-file-ignores]")

with open("pyproject.toml", "w") as f:
    f.write(content)
