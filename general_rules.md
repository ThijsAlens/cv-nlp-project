---
alwaysApply: true
description: General project rules
---

- Always add comments to the code you write. So not only at the top of functions, but also inside the code itself. I do not like large blocks of code without comments.
- The comments should only reference the code itself, it cannot reference our conversation or me directly (so avoid comments like "like you asked").
- Always structure the code in an easy to follow way, using separating comments to visually separate different steps in a file.
- Inside the comments, do not use double backticks \`\` to reference something (they are comments, not a markdown file), so use a single quote character ' instead. So writing for example \`\`marble_calibration\`\` is forbidden.
- Never use the '—' em dash in any of the docs or comments.
- Do not write CLI tools unless explicitly asked for. The preferred project structure uses Python modules which are called from a main runner script.
- For each module (sub)folder, create a README.md explaining that depth of the directory, so you don't need to read all the subfiles to understand what it does.
- Always create a global README.md file and keep it up-to-date with new changes.
- Keep a PROGRESS.md file where you log everything you did. This will help preserve context across chats on everything we did in this project.
