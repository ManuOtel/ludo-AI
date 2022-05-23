import ludopy
import numpy as np
import csv

f = open('/home/manu/Desktop/Manu/SDU/AI/LUDO/game_stats_random5.csv', 'w')
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
        observations = g.get_observation()
        player_i = observations[1]
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner) = observations[0]
        if player_i != 0:
            piece_to_move = random_move(move_pieces)
        else:
            piece_to_move = random_move(move_pieces)
            print(move_pieces)
        new_dice, new_move_pieces, new_player_pieces, new_enemy_pieces, new_player_is_a_winner, there_is_a_winner = g.answer_observation(piece_to_move)
    return g.game_winners[0]


def play_games(number_of_games):
    wins = [0, 0, 0, 0]
    writer.writerow(header)
    for i in range(number_of_games):
        # print('Game -> ' + str(i))
        game_winner = one_game()
        wins[game_winner] += 1
        writer.writerow([str(i), str(game_winner)])
    return wins


if __name__ == '__main__':
    game_number = 1
    stats = play_games(game_number)
    print(stats)
    print('Win rate: ' + str(stats[0] / game_number * 100) + '%')
    f.close()

# print(g.game_winners)
# print(player_pieces)
# print("  ")
# print(enemy_pieces)
# print("Saving history to numpy file")
# g.save_hist(f"game_history.npy")
# print("Saving game video")
# g.save_hist_video(f"game_video.mp4")
