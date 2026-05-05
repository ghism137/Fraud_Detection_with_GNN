---
name: update-session-state
description: Skill for updating the session_state.md file with fine-grained task tracking, timestamps, and strict rules for continuous development.
---

# Update Session State Skill

This skill provides a strict workflow for updating `.claude/project/session_state.md` as the user progresses through continuous development.

## 1. When to use
Use this skill whenever the user asks to "cập nhật tiến độ" (update progress), "chốt phiên" (end session), "log state", or explicitly asks to update `session_state.md`.

## 2. Reading the State
Always read the current `.claude/project/session_state.md` AND `report/sections/REPORT_MAP.md` using `view_file` before making any edits. Do NOT assume their current contents.

## 3. Formatting Rules
When updating `session_state.md`, you must use `multi_replace_file_content` to surgically edit sections, adhering to these rules:

### A. Fine-Grained Task Tracking (Trạng thái tổng thể)
Do not just check off a whole week `[x] Tuần N`. Break down the current week into sub-tasks using an indented list.
- Unstarted task: `- [ ] Task name`
- In-progress task: `- [/] Task name`
- Completed task: `- [x] [YYYY-MM-DD] Task name`

Example transformation:
Before:
`[ ] Tuần 4: KDD & Feature Selection`
After:
```
[/] Tuần 4: KDD & Feature Selection
    - [x] [YYYY-MM-DD] Code pipeline MICE
    - [/] Code thuật toán CART (Information Gain)
    - [ ] Chạy Feature Selection lấy Top 50
```

### B. Timestamping
Always prepend the current date `[YYYY-MM-DD]` when marking a sub-task as done `[x]` or when adding a new Blocker/Decision.

### C. Append-Only (Quyết định đã chốt & Bugs/Blockers)
Never delete historical decisions or bugs. 
- For decisions: Always append to the bottom of the relevant section.
- For bugs/blockers: If a bug is fixed, do not delete it. Apply strikethrough like this: `~~[2026-04-21] Blocker text~~ (Đã fix)`.

### D. Updating Metrics (Kết quả thực nghiệm)
If the user reports a new evaluation metric (AUC-PR, AUC-ROC), locate the markdown table in "Kết quả thực nghiệm" and update the specific cell.

### E. Active Context
Update the "Đang làm" and "File quan trọng đang làm việc" sections to reflect the exact files and immediate task the user is focusing on right now.

### F. Context Map Sync (Đồng bộ REPORT_MAP.md)
If new scripts, notebooks, or resources are created during the session that belong to a specific report section (e.g., a new script for CART should belong to `02_cart`), you MUST update `report/sections/REPORT_MAP.md`.
- Read `REPORT_MAP.md` to find the correct `Section Code`.
- Use `multi_replace_file_content` or `replace_file_content` to append the new file paths to the `Context Files` column of that row.
- Ensure paths are relative (e.g., `src/kdd/new_script.py`).

## 4. Execution
1. Acknowledge the progress conceptually.
2. Read the `session_state.md` and `REPORT_MAP.md` files.
3. Apply edits to `session_state.md` using `multi_replace_file_content`.
4. If new files were added to the project, apply edits to `REPORT_MAP.md` to keep the context routing updated.
5. Respond with a very brief summary of what was logged and mapped.
