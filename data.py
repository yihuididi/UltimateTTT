"""
Augments the dataset created by the CS2109S AY2425 Sem 2 team.

Raw pickle file stored in data.pkl, with 80000 states of Ultimate Tic-Tac-Toe and their respective probability of winning.

Dataset credit: CS2109S AY2425 Sem 2 team
"""

import numpy as np
import pickle
import random
from utils import State, load_data

def inverse_state(state: State) -> State:
    inversed_fill_num = 2 if state.fill_num == 1 else 1
    board = state.board
    inversed_board = np.where(board == 1, 2, np.where(board == 2, 1, board))

    return State(board=inversed_board, fill_num=inversed_fill_num, prev_local_action=state.prev_local_action)

def rotate_state(state: State, k: int) -> State:
    board = state.board

    # Rotate the meta-board (first two dimensions)
    rotated_meta = np.rot90(board, k=k, axes=(0, 1))
    
    # Rotate each local board (last two dimensions)
    rotated_board = np.array([[np.rot90(rotated_meta[i, j], k=k, axes=(0, 1)) 
                               for j in range(3)] for i in range(3)])
    
    # Rotate prev local action if not None
    if state.prev_local_action:
        x, y = state.prev_local_action
        for _ in range(k):
            x, y = 2 - y, x
        rotated_prev_local_action = (x, y)
    else:
        rotated_prev_local_action = state.prev_local_action
    
    return State(board=rotated_board, fill_num=state.fill_num, prev_local_action=rotated_prev_local_action)

def flip_state(state: State, axis: int) -> State:
    if axis < 0 or axis > 1:
        raise ValueError("Axis must be 0 (vertical flip) or 1 (horizontal flip).")
    
    board = state.board
    if axis == 0:
        flipped_meta = np.flip(board, axis=0)
        flipped_board = np.flip(flipped_meta, axis=2)
        if state.prev_local_action:
            x, y = state.prev_local_action
            flipped_prev_local_action = (2 - x, y)
        else:
            flipped_prev_local_action = state.prev_local_action
    else:
        flipped_meta = np.flip(board, axis=1)
        flipped_board = np.flip(flipped_meta, axis=3)
        if state.prev_local_action:
            x, y = state.prev_local_action
            flipped_prev_local_action = (x, 2 - y)
        else:
            flipped_prev_local_action = state.prev_local_action
    
    return State(board=flipped_board, fill_num=state.fill_num, prev_local_action=flipped_prev_local_action)

if __name__ == "__main__":
    data = load_data("data.pkl")
    assert len(data) == 80000

    random.shuffle(data)

    trainset = []
    testset = []
    for i, (state, value) in enumerate(data):
        if state.fill_num == 2:
            state = inverse_state(state)
            value = -value

        assert state.fill_num == 1
        for j in range(4):
            rotated_state = rotate_state(state, j)
            rotated_state_row_data = (rotated_state.board, rotated_state.fill_num, rotated_state.prev_local_action)

            flipped_state = flip_state(rotated_state, 0) # Only flip along 1 axis. Flipping along 2 axis will give duplicates.
            flipped_state_row_data = (flipped_state.board, flipped_state.fill_num, flipped_state.prev_local_action)

            if i < 60000:
                trainset += [(rotated_state_row_data, value), (flipped_state_row_data, value)]
            else:
                testset += [(rotated_state_row_data, value), (flipped_state_row_data, value)]

    print(f"Trainset size = {len(trainset)}, Testset size = {len(testset)}")
    with open("trainset.pkl", "wb") as f:
        pickle.dump(trainset, f)
    with open("testset.pkl", "wb") as f:
        pickle.dump(testset, f)