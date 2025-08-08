import json

# Load the notebook
with open("age-gender.ipynb", "r", encoding="utf-8") as f:
    nb = json.load(f)

# Remove broken widgets metadata if present
if "widgets" in nb.get("metadata", {}):
    del nb["metadata"]["widgets"]

# Save the cleaned notebook
with open("your_notebook_clean.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=2)

print("Cleaned notebook saved as your_notebook_clean.ipynb")
