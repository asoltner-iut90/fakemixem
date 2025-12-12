from io import BytesIO
from google import genai
from google.genai import types
from PIL import Image
from dotenv import load_dotenv
import os


class IA:
    def __init__(self, gemini_api_key=None):
        self.api_key = gemini_api_key if gemini_api_key else os.environ.get("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError("Clé API manquante. Vérifiez .env ou st.secrets.")

        self.client = genai.Client(api_key=self.api_key)

    def generate_image(self, prompt):
        try:
            response = self.client.models.generate_content(
                model="gemini-2.5-flash-image",
                contents=[prompt],
            )

            images = []
            if response.parts:
                for part in response.parts:
                    if part.inline_data:
                        image_data = part.inline_data.data
                        img = Image.open(BytesIO(image_data))
                        images.append(img)
            return images

        except Exception as e:
            print(f"Erreur dans gemini_tools: {e}")
            return []

    def generate_response(self, message, images=None, stream=False):
        if images is None:
            images = []
        contents = [message] + images
        model = "gemini-2.5-flash"

        if stream:
            response = self.client.models.generate_content_stream(
                model=model, contents=contents
            )
            return response
        else:
            response = self.client.models.generate_content(
                model=model, contents=contents
            )
            return response.text


    def get_new_chat(self, functions=[]):
        config = types.GenerateContentConfig(
            tools=functions,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=False)
        )
        return self.client.chats.create(model="gemini-2.5-flash", config=config)

    def send_message(self, message, chat, stream=False):
        if stream:
            response = chat.send_message_stream(message)
            return response
        else:
            response = chat.send_message(message)
            return response.text





if __name__ == "__main__":
    load_dotenv()

    gemini = IA(gemini_api_key=os.environ.get("GOOGLE_API_KEY"))
    for chunk in gemini.generate_response("Explain how AI works.", stream=True):
        print(chunk.text, end="")
