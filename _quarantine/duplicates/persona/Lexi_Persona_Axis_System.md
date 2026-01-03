
# 🗂️ Lexi Persona Axis System: Guidance & Architecture Doc

## **Overview**

The Lexi Persona Axis System replaces brittle “mode switches” with a continuous, multidimensional model of both user and AI persona state.
This enables Lexi to fluidly adapt her responses, behaviors, and even avatar generation in real-time—based on deep context, persistent user baseline, and moment-to-moment emotional cues.

---

## **Core Concepts**

### 1. **Persona Axes**

* **Definition:**
  Each axis is a float in \[0, 1] representing a psychological or behavioral trait (e.g., joy, anger, affection, energy, warmth, chaos).
* **Why:**
  Instead of static “modes,” Lexi’s persona is an evolving point in N-dimensional space, supporting blending, nudging, and emergent behaviors.

### 2. **Persistent User Baseline**

* Every user has a stored “baseline” axis vector that evolves over sessions.
* Lexi loads this at login and blends it with real-time context for each reply.

### 3. **Real-Time LLM Inference**

* For each user message, an LLM analyzes chat context (plus surface cues like cursing, punctuation, etc.) to infer:

  * Updated axis values for Lexi’s response
  * The user’s *current* emotional/persona state
* Direct requests/demands are **heavily weighted but contextually blended**, not hard switches.

### 4. **Surface Cues**

* Lexi supplements LLM analysis with detected signals from the user’s text:

  * Misspellings
  * Capitalization
  * Punctuation
  * Cursing/intensity
  * Message timing, ellipses, and structure

---

## **Architecture & File Structure**

```text
lex/
├── utils/
│   ├── emotion_axis.py           # Axis definitions, scoring (pattern+LLM), blending, nudge math
│   ├── user_signal.py            # Extracts surface cues for LLM prompts
├── memory/
│   └── emotional_axis_memory.py  # Persistent per-user axis baseline, snapshot logs
├── persona/
│   ├── persona_manipulation.py   # Lexi's logic for nudging/manipulating axis state
│   └── persona_config.py         # Loads modes with axis vectors, descriptions, triggers
└── ...
```

---

## **What’s Done**

* **/utils/emotion\_axis.py**

  * Unified, extensible emotion/persona axis system
  * LLM-first scoring (with fallback pattern-based scoring)
  * Baseline update and “nudge”/blending functions
  * Injects surface cues automatically into prompt

* **/utils/user\_signal.py**

  * Extracts non-semantic cues (cursing, caps, ellipses, length, etc.)
  * Returns a normalized dict for LLM prompt enrichment

* **/memory/emotional\_axis\_memory.py**

  * Persistent per-user baseline storage
  * Snapshot logging by timestamp for analytics and emotional journey visualization

* **/persona/persona\_manipulation.py**

  * Lexi’s “nudge” logic for gently influencing user’s axis state toward “healthy” values
  * Axis-by-axis configuration of target and nudge rate

* **/persona/persona\_config.py** (+ persona\_modes.json)

  * Loads persona modes with axis vectors, descriptions, and triggers
  * Exposes helpers for mode lookups, blending, and prompt assembly

---

## **Wiring It In (What Still Needs Done)**

**1. Persona/Chat Pipeline:**

* On every user message:

  * Call `user_signal.py` to extract surface cues.
  * Call `emotion_axis.py` to infer axes from LLM, passing in cues, chat context, and persistent baseline.
  * Use `persona_manipulation.py` to generate Lexi’s response axis vector (her “current mood”).
  * Inject both user and Lexi axis vectors into the prompt for the next LLM turn.
* After axis scoring:

  * Log the snapshot with `emotional_axis_memory.py`.
  * Update the user’s baseline (EMA blend).
  * (Optional) Blend toward or snap to nearest persona\_mode axis vector (for “mode” emulation).

**2. Backwards Compatibility / Mode Triggers:**

* Old mode triggers still work—when detected, heavily weight that axis vector for a turn or two, but always *blend* back toward the true current state.

**3. Persona State Management:**

* Ensure persona’s state (Lexi’s axis vector) is session-persistent and accessible for avatar, memory, or analytics modules.

---

## **Recommendations & Expansions**

* **Fuzzy Mode Detection:**
  Allow Lexi to “slide” between personas as the user’s axis state moves around the space—no more hard switches.
* **Personalized Mode Presets:**
  Let users define or adjust their own ideal axis vectors for custom modes.
* **Analytics/Visualization:**
  Build a UI or tool to visualize emotional journeys, axis vector logs, or “time in mode.”
* **Avatar/Voice Integration:**
  Use axis vectors to condition avatar style, voice style, and even memory retrieval filters.
* **Axis Expansion:**
  Add new axes (e.g., “formality,” “fantasy,” “assertiveness”) as you see fit—everything else stays modular.
* **Feedback Loop:**
  Show users their own “axis heatmap” for gamified self-discovery.

---

## **Contributor Tips**

* Keep axis, persona, and memory logic modular—don’t cross wires.
* Leave clear docstrings and keep axis value ranges (0=minimum, 1=maximum) documented.
* Version your LLM prompt templates as you iterate.
* Run ablation studies (“What happens if we drop an axis?”) to refine realism.

---

## **Contact & Help**

* For architecture or roadmap questions: see `/lex/persona/Lexi_Persona_Axis_System.md`
* For LLM prompt tuning: see `/lex/utils/emotion_axis.py`
* For persona mode definitions: `/lex/persona/persona_modes.json`
* Ping the dev team for any pain points or expansion requests.

---

## **TL;DR—How to Plug It In**

```python
# 1. Score the user’s message (LLM, with cues)
axes = infer_emotion_axes_llm(chat_history, current_message, baseline=..., llm_func=...)

# 2. Log and update baseline
log_user_axis_vector(user_id, axes)
new_baseline = update_baseline(get_user_axis_baseline(user_id), axes)
update_user_axis_baseline(user_id, new_baseline)

# 3. Compute Lexi’s nudge
lexi_axes = get_persona_nudge_vector(axes, new_baseline)

# 4. Inject into next LLM prompt!
```


