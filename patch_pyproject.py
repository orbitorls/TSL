import re
with open("pyproject.toml", "r") as f:
    content = f.read()

content = content.replace("ignore = [", "lint.ignore = [")
content = content.replace("select = [", "lint.select = [")
content = content.replace("[tool.ruff.per-file-ignores]", "[tool.ruff.lint.per-file-ignores]")
content = re.sub(r"\[tool.ruff\]\nlint.ignore =", "[tool.ruff]\n\n[tool.ruff.lint]\nignore =", content)


with open("pyproject.toml", "w") as f:
    f.write(content)
