with open("tests/test_runner.py", "r") as f:
    lines = f.readlines()
for i, line in enumerate(lines):
    if "from src.inference.runner import load_model" in line:
        lines[i] = "from src.inference.runner import TSLRunner\n"
with open("tests/test_runner.py", "w") as f:
    f.writelines(lines)
