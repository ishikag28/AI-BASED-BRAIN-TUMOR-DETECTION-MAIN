import google.generativeai as genai
import json
import os
import mimetypes

genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

def _pick_working_model_name():
    # Allow explicit override first.
    env_model = os.getenv("GEMINI_MODEL")
    if env_model:
        return env_model

    # Try common model ids that support generateContent.
    candidates = [
        "gemini-3-flash-preview",
        "gemini-2.0-flash",
        "gemini-1.5-flash-latest",
        "gemini-1.5-flash",
    ]

    available = set()
    try:
        for m in genai.list_models():
            methods = set(getattr(m, "supported_generation_methods", []) or [])
            if "generateContent" in methods:
                name = getattr(m, "name", "")
                if name:
                    available.add(name.removeprefix("models/"))
    except Exception:
        # If listing models fails, fall back to candidate order.
        return candidates[0]

    for candidate in candidates:
        if candidate in available:
            return candidate

    # Last resort: pick any model that can generate content.
    if available:
        return sorted(available)[0]
    return candidates[0]

model = genai.GenerativeModel(_pick_working_model_name())

def validate_mri(image_path):
    try:
        mime_type, _ = mimetypes.guess_type(image_path)
        if not mime_type or not mime_type.startswith("image/"):
            mime_type = "image/jpeg"

        with open(image_path, "rb") as f:
            image_bytes = f.read()

        prompt = """You are a medical image validator.

Examine the image and determine if it is a brain MRI scan (any modality: T1, T2, FLAIR, etc.).

Return ONLY valid JSON with no extra text:
{
  "isMRI": true or false,
  "reason": "one sentence explanation"
}"""

        response = model.generate_content(
            [prompt, {"mime_type": mime_type, "data": image_bytes}]
        )

        text = response.text.strip()
        text = text.replace("```json", "").replace("```", "").strip()

        data = json.loads(text)
        is_mri = bool(data.get("isMRI", False))
        reason = data.get("reason", "MRI validation completed.")
        return {
            "status": "valid" if is_mri else "invalid",
            "isMRI": is_mri,
            "reason": reason,
        }

    except Exception as e:
        error_text = str(e)
        if "429" in error_text or "quota" in error_text.lower():
            return {
                "status": "unavailable",
                "isMRI": None,
                "reason": "MRI validator unavailable",
            }
        return {
            "status": "unavailable",
            "isMRI": None,
            "reason": f"MRI validator unavailable: {error_text}",
        }