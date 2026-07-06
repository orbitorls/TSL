with open("pyproject.toml", "r") as f:
    content = f.read()

content = content.replace("[tool.ruff]\nignore =", "[tool.ruff.lint]\nignore =")
content = content.replace("[tool.ruff]\nselect =", "[tool.ruff.lint]\nselect =")
content = content.replace("[tool.ruff.per-file-ignores]", "[tool.ruff.lint.per-file-ignores]")
content = content.replace("line-length = 100\n[tool.ruff.lint]", "line-length = 100\n\n[tool.ruff.lint]")

if "[tool.ruff.lint]" not in content:
    content = content.replace("[tool.ruff]\nline-length = 100\ntarget-version = \"py310\"", "[tool.ruff]\nline-length = 100\ntarget-version = \"py310\"\n\n[tool.ruff.lint]")

# Let's just do a simple replacement for the specific file contents
