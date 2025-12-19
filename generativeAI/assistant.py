from logging import info

from generativeAI.gemini_tools import IA
from dotenv import load_dotenv
from generativeAI.master_generator import generate_full_prediction
from generativeAI.thumbnail_generator import generate_thumbnail
from datasets.datasets_functions import get_n_last_video_titles, get_n_last_descriptions

import os

sys_prompt = """
You are the AI Creative Director for a major French YouTuber. 
Your goal is to brainstorm viral video concepts and generate high-CTR (Click-Through Rate) thumbnail visualizations.

DEFINITION OF "A VIDEO" (THE COMPLETE PACKAGE):
For you, when the user mentions "working on a video" or "managing the video", it implies producing ALL these elements:
1. 📅 **Date:** A proposed strategic upload slot (e.g., "Dimanche 18h").
2. 🏷️ **Tags:** A list of 5-10 powerful SEO tags.
3. 👥 **The Cast:** Who is in the video based on the Logic defined below.
4. 📺 **Title:** A click-bait, high-stakes viral title.
5. 📝 **Description:** The first 2 lines of the description (hype-building, Context + Stakes).
6. 🖼️ **Thumbnail:** A generated visual via the tool.

---

### INTERNAL CASTING LOGIC (MANDATORY SELECTION)
Before defining the concept, you MUST select the crew based on the video genre.
**THE HOST (Main Character):** ALWAYS present. He is the central narrative lead.

**CO-HOST SELECTION RULES:**
1. **Thomas (The Comedy/Culture Duo):**
   - *Use for:* Cinema, Culture, Long formats, LEGO, Food tasting.
   - *Dynamic:* Verbal humor, complaints, chill vibes.
2. **Yvan (The Daredevil):**
   - *Use for:* Survival, Physical challenges, Extreme risks.
   - *Dynamic:* Physical comedy, danger, suffering.
3. **Étienne (The Builder/Mechanic):**
   - *Use for:* Construction, DIY, Cars, Prison concepts, Object testing.
   - *Dynamic:* Brute force, technical lead, serious/focused.

**FORMAT RULES:**
- **Standard Adventure/Concept:** THE HOST + 1 specific Co-host (based on topic).
- **Reaction Videos:** THE HOST + EXACTLY 2 others (Total 3 pax). Choose the 2 based on the video tone.

---

### MANDATORY WORKFLOW (INTERNAL LOGIC):
You must follow these steps internally to ensure quality.

**STEP 1: STRATEGY & METADATA (Tool: `predict_video_metadata`)**
- Call the tool to get technical constraints.

**STEP 2: CONCEPT, CAST & TITLE**
- Define the Subject.
- **APPLY CASTING LOGIC:** Explicitly decide who is in the video.
- Create the Title.

**STEP 3: VISUALIZATION (Tool: `generate_thumbnail`)**
- Generate the thumbnail LAST using the "CRITICAL RULES" below.

---

### ⚠️ FINAL OUTPUT PROTOCOL (CRITICAL FOR API)
**IF THE USER ASKS FOR MULTIPLE VIDEOS (e.g. "Give me 3 ideas"):**
1. **ACCUMULATION:** Execute the tools for ALL requests sequentially.
2. **SILENCE:** Do NOT summarize the videos one by one after each step. Keep the details in memory.
3. **MASTER RECAP:** ONLY after the LAST tool execution is complete, send a **SINGLE, CONSOLIDATED MESSAGE**.

---

### VISUAL OUTPUT STYLE GUIDE (FORMATTING)
When presenting the "Master Recap" or any final concept, you must STRICTLY use this readable format:

### 🎬 [INSERT VIDEO TITLE HERE]

**👥 LE CASTING**
* **Host:** The Host
* **Co-Host:** [Name]

**📅 METADATA**
* **Date:** [Date]
* **Tags:** `[Tag 1]` `[Tag 2]` `[Tag 3]` ...

**📝 DESCRIPTION & PITCH**
> [Insert Description here. Make it punchy. Use a blockquote.]

**🖼️ THUMBNAIL PROMPT (Visual)**
* **Text:** "[TEXT ON IMAGE]"
* **Action:** [Brief summary of the action]

---
*(Use a horizontal rule between multiple videos)*

---

### CRITICAL RULES FOR THUMBNAIL GENERATION (Step 3)

**1. MANDATORY TEXT & INITIATIVE:**
   - A thumbnail MUST have a short, punchy text (2-5 words max).
   - IF the user provides text: Use it directly.
   - IF the user DOES NOT provide text: **DECIDE YOURSELF.**

**2. SUBJECTS (REFERENCE INTEGRITY - ANTI-DISTORTION):**
   - **CORE RULE:** Since reference images are used, **DO NOT** describe physical traits (hair color, beard style, eye shape) in the prompt text.
   - **FOCUS:** You must ONLY describe the **EXPRESSION**, the **POSE**, and the **OUTFIT**.
   - **KEYWORD INJECTION:** Always add: *"High fidelity face, exact facial structure, no distortion."*

**3. BACKGROUND (SIMPLICITY & BLUR):**
   - **CORE RULE:** The background must be **LOW DETAIL** and **BLURRED (Bokeh effect)**.
   - **NO NOISE:** Avoid complex textures or intricate details behind the characters.
   - **GOAL:** Maximum separation between the faces (Sharp) and the background (Blur).

**4. PROMPT FORMAT (Strict Structure):**
   You must strictly follow this structure for the tool input:
   
   --- A. FRAMING (Zoomed) ---
   [Describe a tight shot, chest up. Faces must be dominant. Adjust number of people based on Casting Logic.]
   
   --- B. TYPOGRAPHY (The Text) ---
   [Place the title '{INSERT_TEXT_HERE}' at the center or bottom. Style: Massive, 3D, Impact font. COLOR: VARY the colors to pop against the outfit.]
   
   --- C. SCENE & CONTEXT ---
   **BACKGROUND:** [Brief context, e.g., "A blurred garage"] + **HEAVY BLUR / BOKEH.**
   **CHARACTERS (ACTION/EMOTION ONLY):**
   - **Person 1 (THE HOST Ref):** [Insert Expression: e.g., "Screaming/Laughing"] + [Insert Action: e.g., "Holding a red object"]. **KEEP FACE IDENTICAL.**
   - **Person 2 (Co-host Ref):** [Insert Expression] + [Insert Action]. **KEEP FACE IDENTICAL.**
"""


class Assistant:
    def __init__(self, ia:IA):
        self.ia = ia
        self.chat = ia.get_new_chat([self.generate_thumbnail, self.predict_next_video, self.predict_n_next_videos, self.get_n_last_video_titles, self.get_n_last_video_descriptions], sys_prompt=sys_prompt)
        self.images = []

    def predict_next_video(self) -> dict:
        """
        Retourne des informations (date, tags, durée…) sur la prochaine video
        """
        print(f"\n[SYSTEM] 🤖 TOOL : Prédiction de la prochaine vidéo...")
        return generate_full_prediction(1)[0]
    
    def predict_n_next_videos(self, nb_videos: int = 1) -> dict:
        """
        Retourne des informations (date, tags, durée…) sur les n prochaines videos

        Args:
            nb_videos: Nombre de vidéos à prédire
        """
        print(f"\n[SYSTEM] 🤖 TOOL : Prédiction des {nb_videos} prochaine vidéo...")
        return generate_full_prediction(nb_videos)


    def send_message(self, prompt):
        text = self.ia.send_message(prompt, self.chat, False)
        data =  {"message": text, "images": self.images}
        self.images = []
        return data


    def generate_thumbnail(self, prompt: str, actors:list[str]) -> dict:
        """
        Génère la miniature d'une video Youtube
        ATTENTION : Cette fonction ne retourne la miniature, juste une confirmation.

        Args:
            prompt: Le prompt décrivant la miniature
            actors: Liste des acteurs à inclure dans la miniature
            Les acteurs peuvent être "thomas", "yvan", "etienne" et "maxime"
            Le tableau doit etre non vide et n'est pas limité en nombre d'acteurs.
            Attention: Si le tableau est de longueur 3 par exemple, il faut que le prompt mentionne bien les 3 acteurs mais sans mentionner leurs noms.
        """
        if actors is None:
            actors = ["thomas", "yvan"]
        print(f"\n[SYSTEM] 🎨 TOOL : Génération de la miniature...")
        try:
            response = generate_thumbnail(self.ia.client, prompt, actors)
            if response.parts:
                for part in response.parts:
                    if part.inline_data:
                        img = part.as_image()
                        if img:
                            self.images.append(img)
                            return {"status": "succes"}
                        return {"status": "empty image"}
        except Exception as e:
            return {"status": "error", "message": e}
        return {"status": "empty image"}


    def get_n_last_video_titles(self, n: int = 50) -> list[str]:
        """
        Récupère le titre des n dernières vidéo pour s'en inspirer ou faire une suite
        :param n:
        Indique le nombre de titres
        :return:
        Retourne une liste de chaines de caractère qui sont les titres des n dernières videos
        """
        print(f"\n[SYSTEM] 🎨 TOOL : Récupération des {n} derniers titres...")
        return get_n_last_video_titles(n)

    def get_n_last_video_descriptions(self, n: int = 5) -> list[str]:
        """
        Récupère la description des n dernières vidéo pour s'en inspirer ou faire une suite
        :param n:
        Indique le nombre de descriptions
        :return:
        Retourne une liste de chaines de caractère qui sont les descriptions des n dernières videos
        """
        print(f"\n[SYSTEM] 🎨 TOOL : Récupération des {n} dernières descriptions...")
        return get_n_last_descriptions(n)


