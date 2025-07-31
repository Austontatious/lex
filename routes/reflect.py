# friday/routes/reflect.py

from fastapi import APIRouter, Request
from friday.persona import PersonaManager
from friday.model_config import ModelType

router = APIRouter()
persona = PersonaManager()

from lex.memory.memory_core import memory

@router.post("/reflect")
async def reflect_on_huginn(request: Request):
    data = await request.json()
    huginn_output = data.get("idea")
    user_context = data.get("context", "")

    reflection_prompt = (
        "Huginn has imagined the following:\n\n"
        f"{huginn_output}\n\n"
        "Evaluate this speculative idea based on our current conversation context. "
        "Is it insightful, symbolic, or just nonsense? Respond with analysis or emotional framing."
    )

    result = persona.generate(ModelType.FRIDAY, reflection_prompt)

    # ✅ Store as symbolic memory
    memory.store_context(user_context, result["cleaned"])

    return {
        "origin": "friday",
        "analysis_of_huginn": result["cleaned"]
    }

