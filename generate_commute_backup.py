import os
import sys
import json
import time
import asyncio
import datetime
import random
import html
from pathlib import Path
from email.utils import formatdate
from concurrent.futures import ThreadPoolExecutor

import feedparser
import edge_tts
from pydub import AudioSegment
from mutagen.id3 import ID3, CHAP, CTOC, TIT2, CTOCFlags
from google import genai

# ==========================================
# CONFIGURATION & PATHS
# ==========================================
BASE_DIR = Path(__file__).resolve().parent
LIBRARY_DIR = BASE_DIR / "audio_library"
PUBLIC_DIR = BASE_DIR / "public"
EPISODES_DIR = PUBLIC_DIR / "episodes"
STATE_FILE = BASE_DIR / "state.json"

BASE_HOST_URL = os.environ.get("PODCAST_BASE_URL", "http://localhost:8000").rstrip("/")
TTS_VOICE = "en-IN-PrabhatNeural"

# Use model IDs available to your Gemini API project.
# Override without editing the file if needed:
# PowerShell: $env:GEMINI_PRIMARY_MODEL="gemini-3.8-flash"
#             $env:GEMINI_FALLBACK_MODEL="gemini-3.1-pro-preview"
PRIMARY_MODEL = os.environ.get("GEMINI_PRIMARY_MODEL", "gemini-3.8-flash")
FALLBACK_MODEL = os.environ.get("GEMINI_FALLBACK_MODEL", "gemini-3.1-pro-preview")

WEEKDAY_THEMES = {
    0: ("monday", "Moon (Chandra) - Lord Shiva"),
    1: ("tuesday", "Mars (Mangala) - Lord Hanuman & Subramanya"),
    2: ("wednesday", "Mercury (Budha) - Lord Ganesha & Vishnu"),
    3: ("thursday", "Jupiter (Guru) - Lord Dakshinamurthy"),
    4: ("friday", "Venus (Shukra) - Goddess Lakshmi"),
    5: ("saturday", "Saturn (Shani) - Lord Venkateswara"),
    6: ("sunday", "Sun (Surya) - Lord Surya"),
}

RSS_FEEDS = {
    "Bengaluru & Regional": [
        "https://www.thehindu.com/news/cities/bangalore/feeder/default.rss",
        "https://www.deccanherald.com/rss/karnataka.rss",
    ],
    "Indian Markets, Business & IPOs": [
        "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
        "https://economictimes.indiatimes.com/markets/ipos/fpos/rssfeeds/14655708.cms",
        "https://www.livemint.com/rss/markets",
    ],
    "National & Global": [
        "https://www.thehindu.com/news/national/feeder/default.rss",
        "https://www.thehindu.com/news/international/feeder/default.rss",
    ],
    "Sports & Cinema": [
        "https://www.espncricinfo.com/rss/content/story/feeds/0.xml",
        "https://www.thehindu.com/entertainment/feeder/default.rss",
    ],
}

# Shared pool: do not create a new pool for each task.
EXECUTOR = ThreadPoolExecutor(max_workers=4)


# ==========================================
# STATE TRACKER (NON-REPEATING VERSES)
# ==========================================
def load_state() -> dict:
    if STATE_FILE.exists():
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            print("[Notice] state.json could not be read; using initial state.")

    return {
        "chalisa_cycle": 1,
        "next_verse_number": 1,
        "weekday_track_index": {day[0]: 0 for day in WEEKDAY_THEMES.values()},
    }


def save_state(state: dict):
    STATE_FILE.write_text(json.dumps(state, indent=2), encoding="utf-8")


# ==========================================
# RESILIENT GEMINI CALLER
# ==========================================
def run_gemini(client: genai.Client, prompt: str, max_retries: int = 5) -> str:
    """Try configured Gemini models with bounded retries and exponential backoff."""
    models = list(dict.fromkeys([PRIMARY_MODEL, FALLBACK_MODEL]))
    last_error = None

    for attempt in range(max_retries):
        target_model = models[attempt % len(models)]
        try:
            response = client.models.generate_content(
                model=target_model,
                contents=prompt,
            )
            text = getattr(response, "text", None)
            if not text or not text.strip():
                raise RuntimeError(f"{target_model} returned an empty response.")
            return text.strip()
        except Exception as exc:
            last_error = exc
            err = str(exc).lower()
            retryable = any(
                marker in err
                for marker in (
                    "429", "503", "500", "502", "504",
                    "resource_exhausted", "unavailable", "overloaded",
                    "deadline", "timeout", "temporarily",
                )
            )
            if not retryable:
                # Still try the other model once, but don't waste long waits.
                print(f"   [Notice] {target_model} failed: {exc}")
            wait_sec = min(20, 2 ** attempt) + random.uniform(0, 1.0)
            if attempt < max_retries - 1:
                print(
                    f"   [Notice] Gemini request failed on {target_model}; "
                    f"trying again in {wait_sec:.1f}s "
                    f"({attempt + 1}/{max_retries - 1})."
                )
                time.sleep(wait_sec)

    raise RuntimeError(
        "Gemini generation failed after retries. Check API quota, model access, "
        "network connectivity, and GEMINI_API_KEY. Last error: "
        f"{last_error}"
    ) from last_error


# ==========================================
# PARALLEL RSS INGESTION
# ==========================================
def _fetch_single_feed(url: str) -> list[str]:
    entries_text = []
    try:
        parsed = feedparser.parse(url)
        for entry in parsed.entries[:5]:
            title = entry.get("title", "").strip()
            summary = entry.get("summary", "").strip()[:200]
            if title:
                # Strip basic markup before sending RSS content to the model.
                import re
                summary = re.sub(r"<[^>]+>", " ", summary)
                entries_text.append(f"- {title}: {summary}")
    except Exception as exc:
        print(f"   [Notice] RSS feed failed ({url}): {exc}")
    return entries_text


def fetch_rss_parallel() -> str:
    all_urls = [url for urls in RSS_FEEDS.values() for url in urls]
    with ThreadPoolExecutor(max_workers=min(8, len(all_urls) or 1)) as executor:
        results = list(executor.map(_fetch_single_feed, all_urls))

    compiled = [item for sublist in results for item in sublist]
    if not compiled:
        return (
            "No RSS headlines could be retrieved. Do not invent current events, "
            "market levels, IPO subscription figures, traffic updates, or results. "
            "State briefly that live headlines were unavailable."
        )
    return "\n".join(compiled)


# ==========================================
# PROMPT GENERATORS
# ==========================================
def build_chalisa_script(client: genai.Client, start_verse: int, cycle: int) -> tuple[str, int]:
    end_verse = min(start_verse + 1, 43)
    focus = (
        "literal word-by-word meaning and story"
        if cycle == 1
        else f"philosophical depth, discipline, and daily grounding (Cycle {cycle})"
    )

    prompt = f"""
You are a calm, articulate Vedic scholar providing a morning audio reflection on the Hanuman Chalisa.
Cover ONLY Verse {start_verse} and Verse {end_verse} of the 43-part sequence
(Verses 1-2: Opening Dohas; 3-42: Chaupais 1-40; 43: Closing Doha).
Focus: {focus}.

Requirements:
- Target length: approximately 650-750 words.
- Quote each Awadhi verse in clean transliteration for Indian-accent TTS pronunciation.
- Explain key Awadhi words, relevant context, and one practical focus mindset for the workday.
- Output spoken narration only, without markdown, headings, bullets, or stage cues.
"""
    script = run_gemini(client, prompt)
    return script, end_verse


def build_news_script(client: genai.Client, raw_news: str, date_str: str) -> str:
    prompt = f"""
You are a careful radio broadcaster creating a morning commute briefing for {date_str}.
Use the headlines below as the source. Do not invent current news, market values,
IPO subscription figures, grey-market premiums, traffic conditions, or sports results.
If a requested category has no supplied headlines, say that verified headlines were
unavailable for that category rather than guessing.

Create a concise spoken briefing, roughly 1200-1800 words if enough source material exists.
Cover Bengaluru/Karnataka; Indian markets, business and IPOs; national/global technology
and geopolitics; sports and cinema.

Write numbers and abbreviations naturally for text-to-speech. Output narration only,
with no markdown, bullets, headings, stage directions, or sound effects.

Headlines:
{raw_news}
"""
    return run_gemini(client, prompt)


# ==========================================
# AUDIO PIPELINE & ASYNC SYNTHESIS
# ==========================================
async def synthesize_speech(text: str, destination: Path):
    communicate = edge_tts.Communicate(text, TTS_VOICE, rate="+0%")
    await communicate.save(str(destination))


def pick_weekday_track(state: dict, weekday: int) -> tuple[Path, str]:
    folder, label = WEEKDAY_THEMES[weekday]
    folder_path = LIBRARY_DIR / folder
    tracks = sorted(folder_path.glob("*.mp3"))
    if not tracks:
        raise FileNotFoundError(
            f"Missing devotional tracks! Add at least one .mp3 file to: {folder_path}"
        )

    indexes = state.setdefault("weekday_track_index", {})
    idx = indexes.get(folder, 0) % len(tracks)
    chosen = tracks[idx]
    indexes[folder] = (idx + 1) % len(tracks)
    return chosen, label


def stitch_with_chapters(segments: list[tuple[str, Path]], output_file: Path):
    combined = AudioSegment.empty()
    gap = AudioSegment.silent(duration=1500)
    chapter_offsets = []

    for name, file_path in segments:
        if not file_path.exists():
            raise FileNotFoundError(f"Audio segment is missing: {file_path}")
        start_ms = len(combined)
        seg = AudioSegment.from_mp3(file_path)
        combined += seg + gap
        end_ms = len(combined)
        chapter_offsets.append((name, start_ms, end_ms))

    combined.export(output_file, format="mp3", bitrate="128k")

    tags = ID3(output_file)
    elem_ids = []
    for i, (name, s_ms, e_ms) in enumerate(chapter_offsets, 1):
        cid = f"chp{i}"
        elem_ids.append(cid)
        tags.add(
            CHAP(
                element_id=cid,
                start_time=s_ms,
                end_time=e_ms,
                sub_frames=[TIT2(encoding=3, text=[name])],
            )
        )

    tags.add(
        CTOC(
            element_id="toc",
            flags=CTOCFlags.TOP_LEVEL | CTOCFlags.ORDERED,
            child_element_ids=elem_ids,
            sub_frames=[TIT2(encoding=3, text=["Daily Metro Commute"])],
        )
    )
    tags.save(v2_version=3)


def generate_rss(filename: str, title: str, size: int, duration: int):
    safe_title = html.escape(title)
    safe_filename = html.escape(filename, quote=True)
    safe_url = html.escape(f"{BASE_HOST_URL}/episodes/{filename}", quote=True)
    xml_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<rss version="2.0" xmlns:itunes="http://www.itunes.com/dtds/podcast-1.0.dtd">
  <channel>
    <title>Daily Metro Commute</title>
    <link>{html.escape(BASE_HOST_URL)}</link>
    <description>Hanuman Chalisa, Verse Reflections, Devotional Music, and Daily Briefing</description>
    <language>en-in</language>
    <itunes:explicit>false</itunes:explicit>
    <item>
      <title>{safe_title}</title>
      <guid isPermaLink="false">{safe_filename}</guid>
      <pubDate>{formatdate(localtime=True)}</pubDate>
      <enclosure url="{safe_url}" length="{size}" type="audio/mpeg"/>
      <itunes:duration>{duration}</itunes:duration>
    </item>
  </channel>
</rss>"""
    (PUBLIC_DIR / "feed.xml").write_text(xml_content, encoding="utf-8")


# ==========================================
# PARALLEL PIPELINE WORKERS
# ==========================================
async def process_chalisa_meaning(
    client: genai.Client, start_verse: int, cycle: int, output_path: Path
):
    loop = asyncio.get_running_loop()
    print(
        f">> [Pipeline A] Generating Chalisa Meaning "
        f"(Verses {start_verse}-{min(start_verse + 1, 43)})..."
    )
    script, end_verse = await loop.run_in_executor(
        EXECUTOR, build_chalisa_script, client, start_verse, cycle
    )
    print(">> [Pipeline A] Synthesizing Chalisa reflection audio...")
    await synthesize_speech(script, output_path)
    return end_verse


async def process_news_briefing(client: genai.Client, date_str: str, output_path: Path):
    loop = asyncio.get_running_loop()
    print(">> [Pipeline B] Scraping RSS feeds in parallel...")
    raw_news = await loop.run_in_executor(EXECUTOR, fetch_rss_parallel)
    print(">> [Pipeline B] Generating News Briefing script via Gemini...")
    script = await loop.run_in_executor(
        EXECUTOR, build_news_script, client, raw_news, date_str
    )
    print(">> [Pipeline B] Synthesizing News Briefing audio...")
    await synthesize_speech(script, output_path)


# ==========================================
# MAIN EXECUTION
# ==========================================
async def main():
    start_time = time.time()
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: GEMINI_API_KEY environment variable is not set.")
        return

    PUBLIC_DIR.mkdir(parents=True, exist_ok=True)
    EPISODES_DIR.mkdir(parents=True, exist_ok=True)

    client = genai.Client(api_key=api_key)
    state = load_state()

    today = datetime.date.today()
    date_id = today.strftime("%Y_%m_%d")
    date_label = today.strftime("%A, %B %d, %Y")

    track1_path = LIBRARY_DIR / "hanuman_chalisa_mss.mp3"
    if not track1_path.exists():
        raise FileNotFoundError(f"Track 1 missing at {track1_path}")

    track3_path, theme_title = pick_weekday_track(state, today.weekday())

    v_start = int(state.get("next_verse_number", 1))
    v_start = max(1, min(v_start, 43))
    v_cycle = int(state.get("chalisa_cycle", 1))
    track2_path = EPISODES_DIR / f"temp_meaning_{date_id}.mp3"
    track4_path = EPISODES_DIR / f"temp_news_{date_id}.mp3"

    print(
        f">> Launching concurrent AI & Audio pipelines at "
        f"{datetime.datetime.now().strftime('%H:%M:%S')}..."
    )

    tasks = [
        asyncio.create_task(
            process_chalisa_meaning(client, v_start, v_cycle, track2_path)
        ),
        asyncio.create_task(
            process_news_briefing(client, date_label, track4_path)
        ),
    ]

    try:
        results = await asyncio.gather(*tasks)
        end_verse = results[0]
    except asyncio.CancelledError:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise
    except Exception:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise

    final_filename = f"commute_{date_id}.mp3"
    final_mp3 = EPISODES_DIR / final_filename
    segments = [
        ("1. Hanuman Chalisa (M.S. Subbulakshmi)", track1_path),
        (f"2. Chalisa Meaning (Verses {v_start}-{end_verse})", track2_path),
        (f"3. Devotional ({theme_title})", track3_path),
        ("4. Daily India & Global News Briefing", track4_path),
    ]

    print(">> Stitching 4 segments and embedding chapter markers...")
    await asyncio.get_running_loop().run_in_executor(
        EXECUTOR, stitch_with_chapters, segments, final_mp3
    )

    # Update state only after the final episode was successfully assembled.
    state["next_verse_number"] = 1 if end_verse >= 43 else end_verse + 1
    if end_verse >= 43:
        state["chalisa_cycle"] = v_cycle + 1
    save_state(state)

    track2_path.unlink(missing_ok=True)
    track4_path.unlink(missing_ok=True)

    duration_s = int(len(AudioSegment.from_mp3(final_mp3)) / 1000)
    generate_rss(
        filename=final_filename,
        title=(
            f"{today.strftime('%a, %d %b')} - Verses {v_start}-{end_verse} & "
            f"{theme_title.split('-')[0].strip()}"
        ),
        size=final_mp3.stat().st_size,
        duration=duration_s,
    )

    elapsed = round(time.time() - start_time, 1)
    print("\n=======================================================")
    print(f"SUCCESS: Generated {final_mp3.name} in {elapsed}s.")
    print(
        f"Duration: {duration_s // 60} mins | "
        f"Size: {round(final_mp3.stat().st_size / (1024 * 1024), 2)} MB"
    )
    print(f"Feed ready at: {PUBLIC_DIR / 'feed.xml'}")
    print("=======================================================")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nStopped by user (Ctrl+C). Any in-flight Gemini request may take time to stop.")
        sys.exit(130)
    finally:
        EXECUTOR.shutdown(wait=False, cancel_futures=True)
