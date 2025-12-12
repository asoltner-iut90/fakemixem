from google import genai
from google.genai import types
from PIL import Image
import os

# --- Fonction Collage (Inchangée) ---
def create_collage_multi(image_paths_list):
    print(f"--- Création du collage brut ---")
    loaded_images = []
    try:
        for path in image_paths_list:
            loaded_images.append(Image.open(path))
        if not loaded_images: return None
        min_height = min(img.height for img in loaded_images)
        resized_images = []
        for img in loaded_images:
            ratio = min_height / img.height
            new_width = int(img.width * ratio)
            resized_images.append(img.resize((new_width, min_height)))
        total_width = sum(img.width for img in resized_images)
        collage = Image.new('RGB', (total_width, min_height))
        x_offset = 0
        for img in resized_images:
            collage.paste(img, (x_offset, 0))
            x_offset += img.width
        return collage
    except Exception as e:
        print(f"❌ Erreur collage : {e}")
        return None


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    client = genai.Client(api_key=os.getenv("GEN_AI_CLIENT_KEY"))

    titre_video = "100 bêtes sauvages"

    # Vos fichiers
    files = ["yvan.jpg", "thomas.jpg"]
    collage_input = create_collage_multi(files)
    if not collage_input: exit()

    # --- PROMPT AVEC ZOOM OPTIMISÉ ---
    prompt = (
        "You are an expert YouTube thumbnail designer. "
        "Task: Create a high-stakes 'Jungle Survival' movie poster using the people in this reference collage. "
    
        # --- A. CADRAGE ZOOMÉ (Plan Poitrine / Head & Shoulders) ---
        "CRITICAL FRAMING: TIGHT HEAD AND SHOULDERS PORTRAIT. "
        "The camera must be CLOSE to the faces. "
        "Show the men from the top of their heads down to the MIDDLE OF THEIR CHEST only. "
        "Do NOT show the stomach or waist. Zoom in! "
        "Their faces must be large and dominant in the frame. "
    
        # --- B. TYPOGRAPHIE (Ajustée au Zoom) ---
        f"TYPOGRAPHY: Place the title '{titre_video}' at the BOTTOM CENTER. "
        "Since the shot is tight, the text should OVERLAP their chest/shoulders area. "
        "Style: MASSIVE, BOLD, 3D Sans-Serif font (Impact style). "
        "Color: White/Yellow with thick black outline. "
        "Add a dark shadow behind the text so it pops against the shirts. "
    
        # --- C. RESTE DU PROMPT ---
        "COMPOSITION: Remove dividing lines. Unified jungle environment. "
        "IDENTITY: Preserve facial features, hair, and beards exactly. "
        "EXPRESSIONS: Intense, heroic, determined 'Survivor' looks. "
        "ATTIRE: Clean adventurer shirts (Khaki). "
        "BACKGROUND: Dense jungle, ancient ruins, dramatic lighting."
    )

    print(f"\n--- Envoi avec Zoom 'Plan Poitrine' + Titre ---")

    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[prompt, collage_input],
            config=types.GenerateContentConfig(
                image_config=types.ImageConfig(aspect_ratio="16:9"),
                safety_settings=[
                    types.SafetySetting(category="HARM_CATEGORY_HARASSMENT", threshold="BLOCK_ONLY_HIGH"),
                    types.SafetySetting(category="HARM_CATEGORY_DANGEROUS_CONTENT", threshold="BLOCK_ONLY_HIGH"),
                ]
            )
        )

        found = False
        if response.parts:
            for part in response.parts:
                if part.inline_data:
                    print("✅ SUCCÈS ! Miniature zoomée reçue.")
                    img = part.as_image()
                    img.save("resultat_zoom_final.png")
                    img.show()
                    found = True
                elif part.text:
                    print(f"ℹ️ Info : {part.text}")

        if not found:
            print("❌ Echec.")

    except Exception as e:
        print(f"Erreur technique : {e}")