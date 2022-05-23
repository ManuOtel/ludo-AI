import ludopy
import numpy as np
import csv
import torch
import os
from tqdm import tqdm
from deep_q_net import Agent

# from utils import plot_learning_curve

torch.cuda.empty_cache()
f = open(os.getcwd() + '/game_stats_random.csv', 'w')
writer = csv.writer(f)
header = ['number_of_game', 'winner']
g = ludopy.Game()
agent = Agent(gamma=0.99, epsilon=0.5, batch_size=128, action_space=[0, 0, 0, 0], eps_end=0, input_dims=17, lr=0.03)
scores, eps_hist = [], []
avg_score = 0
last_score = 1.0
gama = 0.1


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


def get_action_space(move_pieces):
    action_space = [0, 0, 0, 0]
    for i in move_pieces:
        action_space[i] = 1
    return action_space


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
    #if dice == 6:
    #    reward += 0.01
    if player_is_a_winner:
        reward += 1
    for i in range(4):
        if old_player_pieces[i] == 0 and new_player_pieces[i] != 0:
            reward += 0.5
        if old_player_pieces[i] <= 53 and new_player_pieces[i] >= 54:
            reward += 1
        if old_player_pieces[i] != 59 and new_player_pieces[i] == 59:
            reward += 1
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
        reward = 0
        if player_i != 0:
            piece_to_move = random_move(move_pieces)
            _, _, _, _, _, there_is_a_winner = g.answer_observation(piece_to_move)
        else:
            agent.actions_space = get_action_space(move_pieces=move_pieces)
            # print(move_pieces)
            # print(get_action_space(move_pieces=move_pieces))
            piece_to_move = agent.choose_action(get_state(old_observations[0]))
            # print(move_pieces)
            if (piece_to_move in move_pieces) or (len(move_pieces) == 0 and piece_to_move == -1):
                # print(piece_to_move)
                new_observations = g.answer_observation(piece_to_move)
            else:
                # print(move_pieces)
                # print(agent.actions_space)
                # print(piece_to_move)
                # reward -= 100
                if len(move_pieces) > 0:
                    new_observations = g.answer_observation(move_pieces[0])
                else:
                    new_observations = g.answer_observation(-1)
            # print(move_pieces)
            # print(piece_to_move)
            reward += money_baby(old_observations, new_observations)
            score += reward
            _, _, _, _, _, there_is_a_winner = new_observations
            agent.store_transition(state=get_state(old_observations[0]), action=piece_to_move, reward=reward,
                                   state_=get_state(new_observations), done=there_is_a_winner)
    if g.get_winner_of_game() != 0:
        score += -40
    global last_score
    final_score = score + gama*last_score
    last_score = final_score
    agent.learn()
    scores.append(score)
    eps_hist.append(agent.epsilon)
    avg_score = np.mean(scores[-100:])

    return g.game_winners[0], score, avg_score


def play_games(number_of_game):
    wins = [0, 0, 0, 0]
    writer.writerow(header)
    for i in tqdm(range(0, number_of_game), desc="Current game"):
        # print('Game -> ' + str(i))
        game_winner, score, avg_score = one_game()
        wins[game_winner] += 1
        writer.writerow([str(i), str(game_winner)])
        tqdm.write('Episode: ' + str(i) +
                   ' Score: ' + str(round(score, 2)) +
                   ' Avg_score: ' + str(round(avg_score, 2)) +
                   ' Epsilon: ' + str(round(agent.epsilon, 2)))
    return wins


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    number_of_games = 100
    stats = play_games(number_of_games)
    print('Teaching Win-rate: ' + str(stats[0] / number_of_games * 100) + '%')
    number_of_games = 100
    stats2 = play_games(number_of_games)
    print('True Win-rate: ' + str(stats2[0] / number_of_games * 100) + '%')
    f.close()

# print(g.game_winners)
# print(player_pieces)
# print("  ")
# print(enemy_pieces)
# print("Saving history to numpy file")
# g.save_hist(f"game_history.npy")
# print("Saving game video")
# g.save_hist_video(f"game_video.mp4")
