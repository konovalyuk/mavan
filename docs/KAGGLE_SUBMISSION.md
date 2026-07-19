# Kaggle Capstone Submission

**Competition:** https://www.kaggle.com/competitions/vibecoding-agents-capstone-project  
**Track:** **Agents for Business**  
**Deadline:** verify on the competition page (your timezone)

> Project is still in active development; tag `capstone-v1` when the demo path is stable.

## Checklist before submit

- [ ] Kaggle account + phone verified, rules accepted
- [ ] Public GitHub (no secrets in repo), tag `capstone-v1`
- [ ] Card image 560×280 (AI + Canva or `docs/assets/kaggle_card.png`)
- [ ] YouTube video ≤5 min in media gallery
- [ ] Writeup: track **Agents for Business**, description, GitHub link
- [ ] **Submit** writeup (Save alone is not enough)

---

## 1. GitHub

```bash
git tag capstone-v1
git push origin capstone-v1
```

Do not commit `.env` or API keys.

---

## 2. YouTube video (≤5 min)

### Option A — AI visuals + voiceover (recommended)

1. Generate stills (Leonardo / Ideogram) — prompts in capstone chat storyboard.
2. Animate with Kling / Runway (image → video, ~5–8 s each).
3. Voiceover: ElevenLabs or CapCut TTS — use the shortened Hollywood script (~4:30).
4. Edit in CapCut: 8 scenes, cross-dissolve, auto captions, quiet background music.
5. Upload to YouTube (Unlisted or Public).

### Option B — Screen recording + live demo (strongest for judges)

Record terminal + Swagger (`/docs`) while running commands below. Optional voiceover over B-roll.

**Prep:**

```bash
docker run -d -p 27017:27017 mongo:7   # if needed
source .venv/bin/activate
python main.py
```

`.env`: `CHAT_PROVIDER=gemini`, `GOOGLE_API_KEY`, `GOOGLE_SEARCH_API_KEY`, `AGENT_PYTHON_TOOL=false`

Use [demo rehearsal env](#demo-rehearsal-env) if pipeline returns `insufficient`.

### Option C — Draft slideshow (fastest)

```bash
.venv/bin/python scripts/generate_capstone_assets.py
# → docs/assets/mavan_capstone_demo.mp4, docs/assets/kaggle_card.png
```

Replace with Option A/B before final submit if possible.

---

### Voiceover + screen script (live demo)

| Time | Screen | Say |
|------|--------|-----|
| **0:00–0:40** | Title / problem slide | Decision memory — forecast final outcomes after actions, not generic chat. |
| **0:40–1:30** | CAPSTONE architecture | Stage 1: multi-agent coordinator + specialists, quality, domain model. Stage 2: forecast + recommend. |
| **1:30–2:30** | Terminal | `create_domain.py` → approve sources |
| **2:30–3:30** | Terminal | `run_domain_pipeline.py --provider gemini` → `ready` |
| **3:30–4:30** | Terminal / Swagger | `run_forecast.py --recommend` → show `recommended_action` + probabilities |
| **4:30–5:00** | Features slide | Gemini, tool use, multi-agent, RAG, eval — GitHub link |

### Demo commands (copy-paste)

```bash
python scripts/create_domain.py --name energy --description "Energy policy"

# Runs until Ctrl+C (continuous conveyor)
python scripts/run_domain_pipeline.py --domain-id DOMAIN_ID --provider gemini

python scripts/run_forecast.py --domain-id DOMAIN_ID \
  --state "Oil prices rising" \
  --actions "Increase subsidy,Remove subsidy" --recommend
```

Or via API: `POST /api/v1/domains/{id}/pipeline/start` then later `.../pipeline/stop`.

### Demo rehearsal env

Add to `.env` for local capstone rehearsal (optional):

```env
TRAIN_MIN_SAMPLES=10
TRAIN_MIN_UNIQUE_STATES=3
TRAIN_MIN_UNIQUE_ACTIONS=2
TRAIN_MIN_AVG_QUALITY=60
```

---

## 3. Kaggle Writeup

Overview → **Create Writeup** (discussion posts do not count).

| Field | Value |
|-------|-------|
| Title | MAVAN: Enterprise Decision Intelligence Agents |
| Subtitle | Multi-agent pipeline trains a domain outcome model and fuses it with Gemini for decision forecasts. |
| Track | **Agents for Business** |
| Cover | 560×280 PNG |
| Description | Problem, Stage 1 + 2, agentic features table, Gemini, eval metrics — see [CAPSTONE.md](CAPSTONE.md) |
| GitHub | Public repo URL |
| YouTube | Media gallery link |

Max **2500 words**.

---

## 4. Submit early

Submit before the deadline; certificate badge typically appears on your Kaggle profile after review (~late July 2026).
