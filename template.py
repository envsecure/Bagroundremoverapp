import os

project_name = "bg-remover"

structure = [
    "data/raw",
    "data/processed",
    "data/vocab",

    "configs",

    "src/model",
    "src/data",
    "src/training",
    "src/utils",
    "src/inference",

    "scripts",
    "checkpoints",
    "logs",
]

files = [
    "configs/model.yaml",
    "configs/train.yaml",
    "configs/tokenizer.yaml",

    "src/__init__.py",

    "src/model/__init__.py",
    

    "src/data/__init__.py",
    "src/data/dataset.py",
    "src/data/collate.py",

    "src/training/__init__.py",
    "src/training/trainer.py",
    "src/training/loss.py",
    "src/training/scheduler.py",

    "src/utils/__init__.py",
    "src/utils/checkpoint.py",
    "src/utils/logger.py",
    "src/utils/seed.py",

    "src/inference/generate.py",
    "src/inference/sampling.py",

    "scripts/train.py",
    "scripts/evaluate.py",
    "scripts/inference.py",

    "requirements.txt",
    "README.md",
    ".gitignore",
]

# Create directories
for folder in structure:
    os.makedirs(os.path.join(project_name, folder), exist_ok=True)

# Create files
for file in files:
    file_path = os.path.join(project_name, file)
    if not os.path.exists(file_path):
        open(file_path, "w").close()

print("✅ project structure created successfully!")