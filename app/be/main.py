from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import numpy as np
import pyspiel
import random
from typing import List, Optional, Tuple, Dict
import uuid
import time


from models.bots import play_alphazero, play_ppo


BASE_GAME = None 

@asynccontextmanager
async def lifespan(app: FastAPI):
    global BASE_GAME
    BASE_GAME = pyspiel.load_game("hex(board_size=5)")
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

game_sessions: Dict[str, Dict] = {}


class StartGameRequest(BaseModel):
    model: str = "ppo"  # Default model

class MoveRequest(BaseModel):
    session_id: Optional[str]
    move: List[int]
    model: str
    
class MoveResponse(BaseModel):
    session_id: str 
    move : Optional[List[int]]
    winner: Optional[int]


def cleanup_old_sessions():
    """Remove sessions older than 1 hour"""
    current_time = time.time()
    expired_sessions = [
        session_id for session_id, session_data in game_sessions.items()
        if current_time - session_data.get('created_at', 0) > 3600  # 1 hour
    ]
    for session_id in expired_sessions:
        del game_sessions[session_id]

def get_session(session_id: str):
    """Get session or raise error"""
    cleanup_old_sessions()
    if session_id not in game_sessions:
        raise HTTPException(status_code=404, detail="Session not found or expired")
    return game_sessions[session_id]

def create_new_session(state, model):
    session_id = str(uuid.uuid4())
    game_sessions[session_id] = {
        "state": state,
        "model":model,
        "created_at": time.time()
    }
    return session_id

models = {
    "ppo": play_ppo,
    "ppo_alphazero": play_alphazero
}

def array_to_pyspiel_action(row: int, col: int) -> int:
    """Convert row, col to pyspiel action"""
    return row * 5 + col

def pyspiel_action_to_array(action: int) -> Tuple[int, int]:
    """Convert pyspiel action to row, col"""
    return action // 5, action % 5


@app.post("/move", response_model=MoveResponse)
async def make_move(move_request: MoveRequest):
    if move_request.session_id == None: 
        state = BASE_GAME.new_initial_state()
        model = models.get(move_request.model, None)
        session_id = create_new_session(state, move_request.model)
    else: 
        session_data = get_session(move_request.session_id)
        state = session_data["state"]
        model = models.get(session_data["model"], None)
        session_id = move_request.session_id
        
    if model == None: 
        raise HTTPException(status_code=404, detail="Model not found.")
        
    try: 
        if state.is_terminal():
            raise HTTPException(status_code=400, detail="Game is already over")
        
        action = array_to_pyspiel_action(move_request.move[0], move_request.move[1])
        
        if action not in state.legal_actions():
            raise HTTPException(status_code=400, detail="Illegal move")
        
        state.apply_action(action)
        
        if state.player_reward(0) == 1.0:
            return MoveResponse(
                move=None, 
                winner=1, 
                session_id=session_id
            )
        
        action = model(state)
        
        state.apply_action(action)
        
        move = pyspiel_action_to_array(action)
        
        winner = None 
        
        if state.player_reward(1) == 1.0:
            winner = 2
 
        
        return MoveResponse(
            move=move, 
            winner=winner,
            session_id=session_id
        )
    except HTTPException:
        raise
    except Exception as e:
        print(e)
        raise HTTPException(status_code=500, detail="Something went wrong")


@app.get("/")
async def root():
    return {"message": "Hex Game API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)