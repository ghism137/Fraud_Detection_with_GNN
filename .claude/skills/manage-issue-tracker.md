---
name: manage-issue-tracker
description: Skill for managing architectural decisions, tech debt, and long-term project issues in non_solve_problem.md.
---

# Manage Issue Tracker Skill

This skill provides a standardized workflow for logging and resolving architectural flaws, logical errors, and complex defense questions in `.claude/project/non_solve_problem.md` (Acting as an Architecture Decision Record - ADR).

## 1. When to use
Use this skill when you discover a deep logical flaw in the project architecture, when the user raises a complex academic question, or when you need to track "Tech Debt" that takes multiple sessions to resolve.
Do **NOT** use this for daily progress/small bugs (use `update-session-state` for that).

## 2. Formatting Rules
All issues in `non_solve_problem.md` must follow this structure. Use `multi_replace_file_content` or `replace_file_content` to edit.

### A. Logging a New Issue
Append the new issue to the bottom of the file with the `[OPEN]` tag:
```markdown
### [OPEN] [YYYY-MM-DD] Tên vấn đề ngắn gọn
**Vấn đề/Lỗi Logic:** Mô tả chi tiết tại sao đây là một lỗi hoặc vấn đề.
**Đề xuất/Hành động:** Cách giải quyết hoặc các thực nghiệm cần làm.
```

### B. Resolving an Issue (Append-Only / Strikethrough)
**NEVER delete an issue.** Historical mistakes are valuable for the final thesis defense.
When an issue is solved, apply Markdown strikethrough `~~` to the body text, change the tag to `[SOLVED]`, and add a resolution note.

Example Transformation:
```markdown
### [SOLVED] [2026-04-21] Tên vấn đề ngắn gọn
~~**Vấn đề/Lỗi Logic:** Mô tả chi tiết tại sao đây là một lỗi hoặc vấn đề.~~
~~**Đề xuất/Hành động:** Cách giải quyết hoặc các thực nghiệm cần làm.~~
**Resolution [YYYY-MM-DD]:** Đã giải quyết bằng cách áp dụng phương pháp X.
```

## 3. Execution Flow
1. Read `.claude/project/non_solve_problem.md` using `view_file`.
2. Format the new entry or locate the entry to resolve.
3. Apply edits to the file.
4. Briefly summarize the logged/resolved issue for the user.
