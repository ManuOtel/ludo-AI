import ludopy
import numpy as np
import csv
import os
import torch
from this_one_works import get_action_space

f = open(os.getcwd()+'/game_stats_random5.csv', 'w')
writer = csv.writer(f)
header = ['number_of_game', 'winner']


def random_move(move_pieces):
    if len(move_pieces):
        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
    else:
        piece_to_move = -1
    return piece_to_move


def one_game():
    g = ludopy.Game()
    there_is_a_winner = False
    while not there_is_a_winner:
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner), player_i = g.get_observation()
        if player_i == 0:
            print('Dice = '+str(dice))
            print('Move pieces = '+str(move_pieces))
            print('Action space = '+str(get_action_space(move_pieces)))
            print('Player pieces = '+str(player_pieces))
        if player_i != 0:
            piece_to_move = random_move(move_pieces)
        else:
            piece_to_move = random_move(move_pieces)
            # print((dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
            #        there_is_a_winner), player_i)
        _, _, new_player_pieces, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
        if player_i == 0:
            print('New Player pieces = '+str(new_player_pieces))
            print('Piece to move = '+str(piece_to_move))
            print(" ")
    return g.game_winners[0]


def play_games(number_of_games):
    stats = [0, 0, 0, 0]
    writer.writerow(header)
    for i in range(number_of_games):
        # print('Game -> ' + str(i))
        game_winner = one_game()
        stats[game_winner] += 1
        writer.writerow([str(i), str(game_winner)])
    return stats


if __name__ == '__main__':
    print('cuda' if torch.cuda.is_available() else 'cpu')
    stats = play_games(1)
    print(stats)
    f.close()

# print(g.game_winners)
# print(player_pieces)
# print("  ")
# print(enemy_pieces)
# print("Saving history to numpy file")
# g.save_hist(f"game_history.npy")
# print("Saving game video")
# g.save_hist_video(f"game_video.mp4")
