import ludopy
import numpy as np
import torch
import os
import csv
from tqdm import tqdm
from deep_q_net import Agent

f = open(os.getcwd() + '/game_stats_random.csv', 'w')
writer = csv.writer(f)
header = ['number_of_game', 'winner']

torch.cuda.empty_cache()
g = ludopy.Game()
agent = Agent(gamma=0.75, epsilon=1, batch_size=1024, eps_end=0.1, input_dims=17, lr=0.1)
scores, eps_hist = [], []
avg_score = 0
last_score = 1.0
gama = 0.5


def random_move(move_pieces):
    if len(move_pieces):
        piece_to_move = move_pieces[np.random.randint(0, len(move_pieces))]
    else:
        piece_to_move = -1
    return piece_to_move


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


def money_baby(old_observations, new_observations):
    reward = 0
    (_, old_move_pieces, old_player_pieces, old_enemy_pieces, _, _) = old_observations[0]
    (_, new_move_pieces, new_player_pieces, new_enemy_pieces, player_is_a_winner, _) = new_observations
    if player_is_a_winner:
        reward += 500
    for i in range(4):
        if old_player_pieces[i] == 0 and new_player_pieces[i] != 0:
            reward += 5
        if old_player_pieces[i] != 0 and new_player_pieces[i] == 0:
            reward -= 50
        if old_player_pieces[i] <= 53 and new_player_pieces[i] >= 54:
            reward += 5
        if old_player_pieces[i] != 59 and new_player_pieces[i] == 59:
            reward += 5
        for j in range(3):
            if old_player_pieces[i] != old_enemy_pieces[j][i] and old_enemy_pieces[j][i] != 0 and new_player_pieces[i] == old_enemy_pieces[j][i] and new_enemy_pieces[j][i] == 0:
                reward += 20
    return reward


def get_action_space(move_pieces):
    action_space = [0, 0, 0, 0]
    for i in move_pieces:
        action_space[i] = 1
    return action_space


def one_game(learn=True):
    g.reset()
    score = 0
    there_is_a_winner = False
    while not there_is_a_winner:
        old_observations = g.get_observation()
        player_i = old_observations[1]
        (dice, move_pieces, player_pieces, enemy_pieces, player_is_a_winner,
         there_is_a_winner) = old_observations[0]
        reward = 0
        if player_i != 0:
            piece_to_move = random_move(move_pieces)
            _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
        else:
            piece_to_move = agent.choose_action(get_state(old_observations[0]), get_action_space(move_pieces))
            new_observations = g.answer_observation(piece_to_move)
            reward += money_baby(old_observations, new_observations)
            score += reward
            _, move_pieces_, _, _, _, there_is_a_winner = new_observations
            agent.store_transition(state=get_state(old_observations[0]), action=piece_to_move, reward=reward,
                                   state_=get_state(new_observations), action_space=get_action_space(move_pieces),
                                   action_space_=get_action_space(move_pieces_), done=there_is_a_winner)
    if learn:
        agent.learn()
    global last_score
    final_score = (score + gama * last_score) / 2
    last_score = final_score
    scores.append(score)
    eps_hist.append(agent.epsilon)
    avg_score_now = np.mean(scores[-100:])

    return g.game_winners[0], score, avg_score_now


def play_games(number_of_game, learn=True):
    wins = [0, 0, 0, 0]
    writer.writerow(header)
    for i in tqdm(range(0, number_of_game), desc="Current game"):
        game_winner, score, avg_score_cur = one_game(learn)
        wins[game_winner] += 1
        writer.writerow([str(i), str(game_winner)])
        tqdm.write('Episode: ' + str(i) +
                   ' Score: ' + str(round(score, 2)) +
                   ' Avg_score: ' + str(round(avg_score_cur, 2)) +
                   ' Epsilon: ' + str(round(agent.epsilon, 2)))
    return wins


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    number_of_games = 10000
    stats = play_games(number_of_games, learn=True)
    print('Teaching Win-rate: ' + str(stats[0] / number_of_games * 100) + '%')
    number_of_games = 1000
    agent.epsilon = 0
    stats2 = play_games(number_of_games, learn=False)
    print('True Win-rate: ' + str(stats2[0] / number_of_games * 100) + '%')
    f.close()
