import interface as bbox
import json
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.optimizers import sgd
from keras.models import model_from_json
import copy

# constants:
n_features = 36
n_actions = 4
max_time = -1
epoch = 20
epsilon = .05
max_memory = 36 * 36 * 4
batch_size = 36 * 4
action_repeat = 5
update_frequency = 1000


def prepare_bbox():
    global n_features, n_actions, max_time

    if bbox.is_level_loaded():
        bbox.reset_level()
    else:
        bbox.load_level("../levels/train_level.data", verbose=1)
        n_features = bbox.get_num_of_features()
        n_actions = bbox.get_num_of_actions()
        max_time = bbox.get_max_time()


class ExperienceReplay(object):
    def __init__(self, max_memory=100, discount=.9):
        self.max_memory = max_memory
        self.memory = list()
        self.discount = discount
        self.score_arr = list()

    def remember(self, states, game_over):
        # memory[i] = [[state_t, action_t, reward_t, state_t+1, best_act], game_over?]
        self.memory.append([states, game_over])
        # if len(self.memory) > 200:
        #     self.memory[-1][0][2] = self.memory[-1][0][4] - self.memory[-100][0][4]
        if len(self.memory) > self.max_memory:
            del self.memory[0]

    def get_batch(self, model, batch_size=10):
        len_memory = len(self.memory)
        num_actions = model.output_shape[-1]
        env_dim = self.memory[0][0][0].shape[0]
        inputs = np.zeros((min(len_memory, batch_size), env_dim))
        targets = np.zeros((inputs.shape[0], num_actions))
        for i, idx in enumerate(np.random.randint(0, len_memory, size=inputs.shape[0])):
            state_t, action_t, reward_t, state_tp1, score_t = self.memory[idx][0]
            reward_t -= score_t
            has_next = self.memory[idx][1]
            # if not has_next:  # if has_next is False
            #     targets[i, action_t] = reward_t
            # else:
            #     targets[i, 0:4] = 0.
            #     targets[i, best_act_t] = 1.
            inputs[i] = copy.copy(state_t)
            old_qval = model.predict(state_t.reshape(1, n_features), batch_size=1)
            newQ = model.predict(state_tp1.reshape(1, n_features), batch_size=1)
            maxQ = np.max(newQ)
            targets[i, :] = old_qval[:]
            if has_next:
                update = reward_t + self.discount * maxQ
            else:
                update = reward_t
            targets[i, action] = update
        return inputs, targets


if __name__ == "__main__":
    prepare_bbox()

    train = True
    load_weight = True

    if load_weight:
        model = model_from_json(open('temp_model.json').read())
        model.load_weights('temp_model.h5')
    else:
        model = Sequential()
        model.add(Dense(n_features, init='he_normal', input_shape=(n_features,), activation='relu'))
        model.add(Dense(n_features * 4, init='he_normal', activation='relu'))
        model.add(Dense(2, init='he_normal', activation='relu'))
        model.add(Dropout(0.02))
        model.add(Dense(n_actions, init='he_normal', activation='linear'))
        model.load_weights('model" + str(e) + ".h5')

    model.compile(sgd(lr=.0025), "mse")
    exp_replay = ExperienceReplay(max_memory=max_memory)

    # json_string = model.to_json()
    # model_prim = model_from_json(json_string)
    # model_prim.compile(sgd(lr=.02), "mse")

    if train:
        for e in range(epoch):
            loss = 0.
            bbox.reset_level()
            has_next = 1
            input_t = copy.copy(bbox.get_state())
            score = bbox.get_score()
            tempscore = 0
            update_frequency_cntr = 0
            reset_cntr = 0

            while has_next:
                tempscore = copy.copy(score)
                input_tm1 = copy.copy(input_t)
                if np.random.rand() <= epsilon:
                    action = np.random.randint(0, n_actions, size=1)
                    print(score, bbox.get_time(), loss)

                else:
                    prediction = model.predict(input_tm1.reshape(1, n_features))
                    action = np.argmax(prediction)
                has_next = bbox.do_action(action)
                input_t = copy.copy(bbox.get_state())
                score = copy.copy(bbox.get_score())
                reward = 0
                for i in range(action_repeat):
                    has_next = bbox.do_action(action)
                    reward = copy.copy(bbox.get_score())
                exp_replay.remember([input_tm1, action, reward, input_t, score], has_next)

                inputs, targets = exp_replay.get_batch(model, batch_size=batch_size)

                if bbox.get_time() % 1 == 0:
                    loss = model.train_on_batch(inputs, targets)

            # model_prim.fit(inputs, targets, nb_epoch=1, verbose=0)
            # if update_frequency_cntr >= update_frequency:
            #     prim_weights = model_prim.get_weights()
            #     print("\nmodel update")
            #     print(score, action, best_act, bbox.get_time(), '\n\n')
            #     model.set_weights(prim_weights)
            #     update_frequency_cntr = 0
            # update_frequency_cntr += 1
            # if reward < -300:
            #     bbox.reset_level()
            #     reset_cntr += 1

            print("\n\nEpoch {:03d}/999 | Score {}\n\n".format(e, bbox.get_score()))

            json_string = model.to_json()
            open("model" + str(e) + ".json", 'w').write(json_string)
            model.save_weights('model' + str(e) + '.h5')
    else:
        has_next = 1
        while has_next:
            input_tm1 = copy.copy(bbox.get_state())
            prediction = model.predict(input_tm1.reshape(1, n_features))
            action = np.argmax(prediction)
            has_next = bbox.do_action(action)
        bbox.finish(verbose=1)
