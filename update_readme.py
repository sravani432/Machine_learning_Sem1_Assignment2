import pandas as pd
import json

# Load metrics from JSON file
with open("results.json", "r") as f:
    results = json.load(f)

# Convert to DataFrame
df = pd.DataFrame(results).T

# Build Markdown table manually (no tabulate dependency)
markdown_table = "| Model | " + " | ".join(df.columns) + " |\n"
markdown_table += "|-------|" + "|".join(["-------"] * len(df.columns)) + "|\n"

for model, row in df.iterrows():
    markdown_table += f"| {model} | " + " | ".join(f"{val:.4f}" for val in row) + " |\n"

# Update README
readme_path = "README.md"
with open(readme_path, "r", encoding="utf-8") as f:
    content = f.read()

start_marker = "## Comparison of Models and Metrics"
end_marker = "## Observations on Model Performance"

start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    new_content = (
        content[:start_idx]
        + start_marker
        + "\n\n"
        + markdown_table
        + "\n\n"
        + content[end_idx:]
    )

    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(new_content)

    print("✅ README.md updated with latest metrics.")
else:
    print("⚠️ Could not find placeholder section in README.md.")
