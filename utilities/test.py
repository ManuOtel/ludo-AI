import ludopy
import numpy as np

g = ludopy.Game()
there_is_a_winner = False


def get_state(dice, move_pieces, player_pieces, enemy_pieces):
    state = [dice / 6]
    for i in move_pieces:
        state.append(i / 3)
    for i in player_pieces:
        state.append(i / 59)
    for i in enemy_pieces:
        for j in i:
            state.append(j / 59)
    return state


while not there_is_a_winner:
    (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner), player_i = g.get_observation()

    if len(move_pieces):
        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
        print(move_pieces)
    else:
        piece_to_move = -1

    dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner, there_is_a_winner = g.answer_observation(piece_to_move)


state = get_state(dice, move_pieces, player_pieces, enemy_pieces)


#print('Dice: ' + str(dice/6))
#print('Move Pieces: ' + str(move_pieces/3))
#print('Player Pieces: ' + str(player_pieces/59))
#print('Enemy Pieces: ' + str(enemy_pieces/59))
#print('Player is winner: ' + str(int(player_is_a_winner)))
#print('There is winner: ' + str(int(there_is_a_winner)))

print(state)