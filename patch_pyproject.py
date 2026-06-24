import sys

def modify_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()

    # Apply replacement
    if filepath == 'pyproject.toml':
        # Move ruff settings
        search = """[tool.ruff]
target-version = "py310"
line-length = 100
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "ARG", # flake8-unused-arguments
    "SIM", # flake8-simplify
]
ignore = [
    "E501",  # line too long (handled by black)
    "B008",  # do not perform function calls in argument defaults
    "B904",  # raise without from inside except
]

[tool.ruff.per-file-ignores]
"__init__.py" = ["F401"]  # unused imports
"train_tsl51_v3.py" = ["E402"]  # module level import not at top"""

        replace = """[tool.ruff]
target-version = "py310"
line-length = 100

[tool.ruff.lint]
select = [
    "E",   # pycodestyle errors
    "W",   # pycodestyle warnings
    "F",   # pyflakes
    "I",   # isort
    "B",   # flake8-bugbear
    "C4",  # flake8-comprehensions
    "UP",  # pyupgrade
    "ARG", # flake8-unused-arguments
    "SIM", # flake8-simplify
]
ignore = [
    "E501",  # line too long (handled by black)
    "B008",  # do not perform function calls in argument defaults
    "B904",  # raise without from inside except
]

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]  # unused imports
"train_tsl51_v3.py" = ["E402"]  # module level import not at top"""

        if search in content:
            content = content.replace(search, replace)
            print("Patched pyproject.toml")
        else:
            print("Could not find search block in pyproject.toml")

    with open(filepath, 'w') as f:
        f.write(content)

modify_file('pyproject.toml')
