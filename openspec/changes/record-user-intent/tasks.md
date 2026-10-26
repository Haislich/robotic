# Tasks

## 1. Intent document

- [x] 1.1 Install `intent.md` of this change as `openspec/intent.md`, after the user approves its text (design D1); verify the file matches the approved text

## 2. OpenSpec configuration

- [x] 2.1 Point `context` in `openspec/config.yaml` to `openspec/intent.md` and add the proposal and design rules of design D2, after the user approves the wording; verify the file parses as YAML and `context` stays a string
- [x] 2.2 Add the apply and archive guidance of design D3, after the user approves the wording; verify the file parses as YAML

## 3. Working contract

- [x] 3.1 Replace the Roles section of `CLAUDE.md` with the text of design D4 and add the `openspec/intent.md` row to the documentation homes table, after the user approves; verify "Do not decide on the user's behalf" no longer appears
- [x] 3.2 Add `openspec/intent.md` to the paths table of `README.md`; verify the table names it

## 4. Memory

- [x] 4.1 Rewrite and update the memory files per design D5; verify `memory/MEMORY.md` lists every file in `memory/` and no memory states that the numeric side is secondary
