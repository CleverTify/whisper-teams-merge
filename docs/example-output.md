# Example output

What `transcript.md` looks like, so you can see the shape before committing an
hour and 25 GB to finding out.

**The dialogue below is invented.** Real output from this project is not in the
repository and never will be — it is a recording of people talking, and that is
not ours to publish. Everything else here (the header fields, the screen
annotations, the per-turn language tags) is exactly what the exporter writes.

---

```markdown
# Team sync-20260804_173547-Meeting Recording.mp4

- **Duration:** 00:41:18
- **Languages:** cs 94%, en 6%
- **Speakers:** 2 (from teams)
- **ASR:** faster-whisper / large-v3
- **LLM merge:** 27 turns improved, 3 rejected by guardrail

---

> *screen [00:00:04]:* A kanban board with four columns — Backlog, In progress,
  Review, Done. The Review column holds two cards.

## [00:00:07] Jan Novák  `cs`

Tak jo, už by to mělo nahrávat. Můžeme začít.

## [00:00:11] Petra Dvořáková

Já jsem se chtěla zeptat na ten deployment. Pustíme to ve středu, nebo
počkáme na review?

## [00:00:19] Jan Novák

Počkáme. Ještě tam mám dvě věci v tom sloupci a nechci to pouštět rozpracované.

> *screen [00:01:02]:* The same board, now filtered to one assignee. A card
  titled "Migrace databáze" has been dragged from Review to Done.

## [00:01:05] Petra Dvořáková

Tak tuhle můžeme zavřít, ta je hotová.

## [00:02:41] Jan Novák  `en`

Let's keep the rest in Czech, I'll just say this bit in English for the notes.
```

---

## The other three files

`transcript.vtt` — the same turns as word-timed subtitles with speaker tags,
which is what you drop into a video player:

```
WEBVTT

00:00:07.120 --> 00:00:09.480
<v Jan Novák>Tak jo, už by to mělo nahrávat.
```

`result.json` — every word with its time, speaker and language, plus the full
provenance of the run: which models, which settings, every line the LLM changed
and every line its guardrail rejected, and the hardware it ran on. This is the
file to read if you want to know *why* a transcript says what it says.

`removed.jsonl` — one line per segment the cleanup stage dropped, with the
reason. Subtitle boilerplate ("Titulky vytvořil…"), repetition loops, and
segments too short to be speech. An audit log, so nothing disappears silently.

## Naming

Folder names are slugged from the source file, so
`Team sync-20260804_173547-Meeting Recording.mp4` becomes
`output/team-sync-20260804-173547-meeting-recording/`. The original name is kept
in `result.json` and shown in the UI.
