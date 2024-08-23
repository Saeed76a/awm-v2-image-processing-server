from pydantic import BaseModel
from fastapi import APIRouter, HTTPException, Depends
from API_server import APIServer
from dependencies import get_api_server
from typing import List

class ChattingRequest(BaseModel):
    location_info: List[str]
    query: str
class ChattingResponse(BaseModel):
    bot_response: str
router = APIRouter()

@router.post("/chattingbot/", response_model=ChattingResponse, tags=["chattingbot"])
async def chatting_bot(user_data: ChattingRequest, server: APIServer = Depends(get_api_server)):
    try:
        ai_client = server.chatbot_client

        query = user_data.query
        information = user_data.location_info

        completion = ai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system", 
                    "content": 
                        f"""
                            당신은 기존의 받은 정보를 가지고 질문이 들어오면 적절한 대답을 해주는 AI assistant 입니다.
                            기존 정보는 다음과 같습니다.
                            {', '.join(information)}
                        """
                    },
                {
                    "role": "user",
                    "content": f"{query}"
                }
            ]
        )
        response = completion.choices[0].message.content
        return ChattingResponse(bot_response=response)
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


