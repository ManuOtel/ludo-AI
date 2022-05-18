import ludopy
import numpy as np
import csv
import torch
from deep_q_net import Agent

# from utils import plot_learning_curve

f = open('/home/manu/Desktop/Manu/SDU/AI/LUDO/game_stats_random.csv', 'w')
writer = csv.writer(f)
header = ['number_of_game', 'winner']
g = ludopy.Game()
agent = Agent(gamma=0.99, epsilon=1.0, batch_size=64, n_actions=4, eps_end=0.01, input_dims=[17], lr=0.003)
scores, eps_hist = [], []
avg_score = 0


def get_state(observations):
    (dice, _, player_pieces, enemy_pieces, _, _) = observations
    index = 1
    state = np.zeros(17)
    state[index] = dice
    for i in player_pieces:
        state[index] = i
        index += 1
    for i in enemy_pieces:
        for j in i:
            state[index] = j
            index += 1
    return state


def random_move(move_pieces):
    if len(move_pieces):
        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
    else:
        piece_to_move = -1
    return piece_to_move


def strategy_move(move_pieces):
    if len(move_pieces):
        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
    else:
        piece_to_move = -1
    return piece_to_move


def money_baby(old_observations, new_observations):
    reward = 0
    (dice, old_move_pieces, old_player_pieces, old_enemy_pieces, _, _) = old_observations[0]
    (_, new_move_pieces, new_player_pieces, new_enemy_pieces, player_is_a_winner, _) = new_observations
    if dice is 6:
        reward += 0.01
    if player_is_a_winner:
        reward += 10
    for i in range(4):
        if old_player_pieces[i] == 0 and new_player_pieces[i] != 0:
            reward += 0.5
        if old_player_pieces[i] <= 53 and new_player_pieces[i] >= 54:
            reward += 1
        if old_player_pieces[i] != 59 and new_player_pieces[i] == 59:
            reward += 2
        for j in range(3):
            if old_enemy_pieces[j][i] != 0 and new_enemy_pieces[j][i] == 0:
                reward += 0.8

    return reward


def one_game():
    g.reset()
    score = 0
    there_is_a_winner = False
    while not there_is_a_winner:
        old_observations = g.get_observation()
        player_i = old_observations[1]
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner) = old_observations[0]
        if player_i != 0:
            piece_to_move = random_move(move_pieces)
            _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
        else:
            piece_to_move = agent.choose_action(get_state(old_observations[0]))
            #print(move_pieces)
            if piece_to_move in move_pieces:
                #print(piece_to_move)
                new_observations = g.answer_observation(piece_to_move)
            else:
                if len(move_pieces) > 0:
                    new_observations = g.answer_observation(move_pieces[0])
                else:
                    new_observations = g.answer_observation(-1)
            reward = money_baby(old_observations, new_observations)
            score += reward
            _, _, _, _, _, there_is_a_winner = new_observations
            agent.store_transition(state=get_state(old_observations[0]), action=piece_to_move, reward=reward,
                                   state_=get_state(new_observations), done=there_is_a_winner)
    if g.get_winner_of_game() is not 0:
        score += -40
    agent.learn()
    scores.append(score)
    eps_hist.append(agent.epsilon)
    avg_score = np.mean(scores[-100:])

    return g.game_winners[0], score, avg_score


def play_games(number_of_game):
    wins = [0, 0, 0, 0]
    writer.writerow(header)
    for i in range(number_of_game):
        # print('Game -> ' + str(i))
        game_winner, score, avg_score = one_game()
        wins[game_winner] += 1
        writer.writerow([str(i), str(game_winner)])
        print('Episode: ', i, ' Score %.2f' % score, ' Avg_score %.2f' % avg_score, ' Epsilon %.2f' % agent.epsilon)
    return wins


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    number_of_games = 10000
    stats = play_games(number_of_games)
    print('Win-rate: ' + str(stats[0] / number_of_games * 100) + '%')
    f.close()

# print(g.game_winners)
# print(player_pieces)
# print("  ")
# print(enemy_pieces)
# print("Saving history to numpy file")
# g.save_hist(f"game_history.npy")
# print("Saving game video")
# g.save_hist_video(f"game_video.mp4")
