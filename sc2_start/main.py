import sc2
from sc2 import run_game, maps, Race, Difficulty, position
from sc2.player import Bot, Computer
from sc2.constants import NEXUS, PROBE, PYLON, ASSIMILATOR, GATEWAY, \
 CYBERNETICSCORE, STARGATE, VOIDRAY, ZEALOT, STALKER, ROBOTICSFACILITY, \
OBSERVER, IMMORTAL, ADEPT, FORGE, SHIELDBATTERY, PHOTONCANNON, TWILIGHTCOUNCIL, \
DARKSHRINE, DARKTEMPLAR, ORBITALCOMMAND, COMMANDCENTER, DESTRUCTIBLEROCK2X4VERTICAL, \
DESTRUCTIBLEROCK2X4HORIZONTAL, DESTRUCTIBLEROCK2X6VERTICAL, DESTRUCTIBLEROCK2X6HORIZONTAL, \
DESTRUCTIBLEROCK4X4, DESTRUCTIBLEROCK6X6
import multiprocessing
import random
import asyncio
import numpy as np
import pickle
import datetime
import glob
import operator
import os
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, MinMaxScaler, LabelBinarizer
import lightgbm as lgb
import traceback
from keras import layers, models, callbacks
import tensorflow
import h5py
import pickle
import time
import pandas as pd

max_iter = 1000000
class_num = 30
var_size = 41
memory_size = 500
nan_replacement = -1
max_game_training_size = 1000
perc_to_consider = .1


aggressive_units = {
    ZEALOT: [8, 3],
    STALKER: [8, 3],
    IMMORTAL: [8, 3],
    VOIDRAY: [8, 3],
    ADEPT: [8, 3],
    DARKTEMPLAR: [8, 3],
}
dump_dict = dict()

path = r'C:\Users\trist\Documents\sc2_bot/'
record_memory = 2000


def train_strat_model():
    files = glob.glob(path + '*_data.plk')
    dicts = []

    # if len(files) > max_game_training_size:
    #     files = random.sample(files, max_game_training_size)

    for i in files:
        with open(i, 'rb') as f:
            dicts.append(pickle.load(f))

    random.shuffle(dicts)
    dicts.sort(reverse = True, key = lambda x: x['score'])
    print(len(dicts))
    print([i['score'] for i in dicts])

    features = sum([i['past_moves'] for i in dicts], [])
    # random.shuffle(features)
    x = np.array([i['game_state'] for i in features])
    y = np.array([i['f'] for i in features])

    # x = x.reshape((-1, var_size * memory_size))
    x = np.squeeze(x)
    scaler = MinMaxScaler()
    scaler.fit(x)
    enc = LabelBinarizer(sparse_output = False)
    enc.fit(np.reshape(y, (-1, 1)))

    with open(path + 'scaler.plk', 'wb') as f:
        pickle.dump(scaler, f)
    with open(path + 'encoder.plk', 'wb') as f:
        pickle.dump(enc, f)

    pos_dicts = dicts[:int(len(dicts)*perc_to_consider)]
    neg_dicts = dicts[-int(len(dicts)*perc_to_consider):]
    pos_features = sum([i['past_moves'] for i in pos_dicts], [])
    neg_features = sum([i['past_moves'] for i in neg_dicts], [])

    pos_x = np.array([i['game_state'] for i in pos_features])
    neg_x = np.array([i['game_state'] for i in neg_features])
    pos_y = np.array([i['f'] for i in pos_features])
    neg_y = np.array([i['f'] for i in neg_features])

    # pos_x = pos_x.reshape((-1, var_size * memory_size))
    # neg_x = neg_x.reshape((-1, var_size * memory_size))
    pos_x = np.squeeze(pos_x)
    neg_x = np.squeeze(neg_x)

    pos_x = scaler.transform(pos_x)
    neg_x = scaler.transform(neg_x)

    pos_y = enc.transform(np.reshape(pos_y, (-1, 1)))
    neg_y = enc.transform(np.reshape(neg_y, (-1, 1)))

    # x = np.vstack([pos_x, neg_x])
    pos_y = pos_y.astype(np.float32)
    neg_y = (neg_y == 0).astype(np.float32)
    # y = np.vstack([pos_y, neg_y])
    x = pos_x
    y = pos_y
    # model = ExtraTreesClassifier(n_jobs=-1, min_samples_leaf = 50)
    # model.fit(pos_x, pos_y)
    #
    # with open(path + 'model.plk', 'wb') as f:
    #     pickle.dump(model, f)
    #
    # res = model.predict(x)
    # print(res)
    #


    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=.1)
    #
    cb = [callbacks.EarlyStopping(patience=0),
                callbacks.ModelCheckpoint(path + 'dnn.h5',
                                save_best_only=True,
                                save_weights_only=False)]

    model = models.Sequential()
    model.add(layers.Dense(2000, input_dim=45 + memory_size, activation='elu'))
    model.add(layers.Dense(2000, activation='elu'))
    model.add(layers.Dense(2000, activation='elu'))

    model.add(layers.Dense(class_num, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['mae'])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), callbacks=cb, epochs=100, batch_size=512)


def get_closest(unit, unit_list):
    closest_unit = None
    closest_distance = 999

    for i in unit_list:
        if i.distance_to(unit) < closest_distance:
            closest_distance = i.distance_to(unit)
            closest_unit = i
    return closest_unit


def get_closest_distance(units, e_units):
    closest_distance = 999

    for i in units:
        for j in e_units:
            if i.distance_to(j) < closest_distance:
                closest_distance = i.distance_to(j)
    return closest_distance


class Strat():
    random_chance = .01
    print('random', random_chance)
    def __init__(self):
        try:
            self.model = models.load_model(path + 'dnn.h5')
            # with open(path + 'model.plk', 'rb') as f:
            #     self.model = pickle.load(f)

            with open(path + 'scaler.plk', 'rb') as f:
                self.scaler = pickle.load(f)
            with open(path + 'encoder.plk', 'rb') as f:
                self.encoder = pickle.load(f)
            self.use_model = True
        except:
            traceback.print_exc()
            self.use_model = False

    def get_move(self, move_dict, game_state):
        if random.random() < self.random_chance or not self.use_model:
            next_move = random.choice(move_dict)
        else:
            game_state = np.expand_dims(game_state, 0)
            # print(game_state.shape)
            game_state = np.reshape(game_state, (1, -1))
            # print(game_state.shape)

            # game_state = game_state.reshape((-1, var_size * memory_size))
            scaled_input = self.scaler.transform(game_state)
            a = self.model.predict(scaled_input)
            # next_move = a[0]
            # print(next_move)
            p = a[0]
            p /= p.sum()

            # print(p.tolist())
            next_move_index = np.random.choice(np.array([i for i in range(p.shape[0])]), p = p)
            next_move_array = [0 for _ in range(p.shape[0])]
            next_move_array[next_move_index] = 1
            next_move_array =np.array([next_move_array])
            next_move = self.encoder.inverse_transform(np.array(next_move_array))[0]
            # print(next_move)
            # next_move = np.argmax(a[0])
        return next_move


class SentdeBot(sc2.BotAI):

    def __init__(self, s):
        super().__init__()
        self.ts = int(datetime.datetime.now().timestamp())
        self.ITERATIONS_PER_MINUTE = 60
        self.MAX_WORKERS = 100
        self.memory = memory_size
        self.actions = [i for i in range(class_num)]
        self.past_moves = []
        self.s = s
        self.max_score = 0
        self.games_states = []
        self.move_history = []
        self.counter = 0

        for i in range(memory_size + 1):
            self.games_states.append([nan_replacement for j in range(var_size - 1)])
            self.move_history.append(-1)


    def run_reward_func(self):
        # return len([i for i in self.units if i.name == 'Voidray'])
        # return self.state.score.score + (self.state.score.killed_value_units * 5) - self.minerals - self.vespene
        # return ( self.state.score.score - self.minerals - self.vespene) + (self.state.score.killed_value_units * ( self.state.score.score - self.minerals - self.vespene))
        # return self.state.score.score
        # return self.state.score.killed_value_units + (self.state.score.killed_value_structures * self.state.score.killed_value_structures)
        self.max_score =  max(self.state.score.killed_value_units, self.max_score)


    def get_state(self):
        d1 = self.distance_nexus_to_enemy()
        d2 = self.distance_to_enemy()
        d3 = self.distance_to_enemy_buildings()

        if not d1:
            d1 = -1
        if not d2:
            d2 = -1
        if not d3:
            d3 = -1

        dead_unit_count = len(self.state.dead_units)
        unit_count = len(self.units)
        game_loop = self.state.game_loop
        minerals = self.minerals
        vg = self.vespene
        l_vg = len(self.geysers)
        n_probes = len(self.workers)
        supply_cap = self.supply_cap
        supply_left = self.supply_left
        supply_used = self.supply_used

        num_th = len(self.townhalls)
        keu = len(self.known_enemy_units)
        kes = len(self.known_enemy_structures)

        n_gates = len([ i for i in self.units if i.name == 'Gateway'])
        n_cyber = len([ i for i in self.units if i.name == 'CyberneticsCore'])
        n_stalker = len([ i for i in self.units if i.name == 'Stalker'])
        n_stargate = len([ i for i in self.units if i.name == 'Stargate'])
        n_void = len([ i for i in self.units if i.name == 'Voidray'])
        n_robo = len([ i for i in self.units if i.name == 'RoboticsFacility'])
        n_im = len([ i for i in self.units if i.name == 'Immortal'])
        n_zealot = len([ i for i in self.units if i.name == 'Zealot'])
        n_obs= len([ i for i in self.units if i.name == 'Observer'])
        n_adept= len([ i for i in self.units if i.name == 'Adept'])
        n_dt = len([i for i in self.units if i.name == 'DarkTemplar'])
        n_cannon = len([i for i in self.units if i.name == 'PhotonCannon'])
        n_sheild = len([i for i in self.units if i.name == 'ShieldBattery'])
        probe_by_nexus = n_probes/max(1, num_th)
        probe_by_units = n_probes/max(1, supply_used)

        n_gates_e = len([ i for i in self.known_enemy_units if i.name == 'Gateway'])
        n_cyber_e = len([ i for i in self.known_enemy_units if i.name == 'CyberneticsCore'])
        n_stalker_e = len([ i for i in self.known_enemy_units if i.name == 'Stalker'])
        n_stargate_e = len([ i for i in self.known_enemy_units if i.name == 'Stargate'])
        n_void_e = len([ i for i in self.known_enemy_units if i.name == 'Voidray'])
        n_robo_e = len([ i for i in self.known_enemy_units if i.name == 'RoboticsFacility'])
        n_im_e = len([ i for i in self.known_enemy_units if i.name == 'Immortal'])
        n_zealot_e = len([ i for i in self.known_enemy_units if i.name == 'Zealot'])
        n_obs_e= len([ i for i in self.known_enemy_units if i.name == 'Observer'])
        n_adept_e= len([ i for i in self.known_enemy_units if i.name == 'Adept'])
        n_dt_e = len([i for i in self.known_enemy_units if i.name == 'DarkTemplar'])
        n_cannon_e = len([i for i in self.known_enemy_units if i.name == 'PhotonCannon'])
        n_sheild_e = len([i for i in self.known_enemy_units if i.name == 'ShieldBattery'])

        return [dead_unit_count, unit_count, game_loop, minerals, supply_cap, supply_left, num_th, keu, kes,
                vg, l_vg, n_probes, n_gates, n_cyber, n_stalker,n_zealot,n_stargate,n_void,n_robo,n_im,
                n_gates_e, n_cyber_e, n_stalker_e, n_stargate_e, n_void_e, n_robo_e, n_im_e, n_zealot_e, n_obs_e, n_adept_e, n_dt_e, n_cannon_e, n_sheild_e, n_obs, n_adept, n_dt,
                n_cannon, n_sheild, d1, d2, d3, supply_used, probe_by_nexus, probe_by_units,
                self.counter]


    async def on_step(self, iteration):
        global dump_dict

        self.counter += 1
        self.iteration = iteration
        try:
            # print(self.time)
            # if self.time > 600:
            #     return

            self.run_reward_func()

            dump_dict = {'score':self.max_score, 'past_moves':self.past_moves}

            game_state = self.get_state()
            self.games_states.append(game_state)
            t_game_state = self.games_states[-1]
            np_t_game_state = np.expand_dims(np.array(t_game_state), 0)
            np_t_past_moves = np.expand_dims(np.array(self.move_history[-memory_size:]), 0)
            np_t_game_state = np.hstack([np_t_game_state, np_t_past_moves])

            f = self.actions[self.s.get_move(self.actions, np_t_game_state)]

            # #artificial fixes
            if len(self.townhalls) < 2 and random.random() < .8:
                f = 4
            while ((f == 2 or f == 28) and (self.supply_cap > 190 or self.supply_cap/len(self.townhalls) > 25)):
                f = self.actions[self.s.get_move(self.actions, np_t_game_state)]

            self.move_history.append(f)

            if f == 0:
                await  self.distribute_workers()
            if f == 1:
                await  self.build_workers()
            if f == 2:
                await  self.build_pylons()
            if f == 3:
                await  self.build_assimilators()
            if f == 4:
                await  self.expand()
            if f == 5:
                await  self.build_gateway()
            if f == 6:
                await  self.build_zealot()
            if f == 7:
                await  self.attack()
            if f == 8:
                await  self.build_stalker()
            if f == 9:
                await  self.build_cybernetic()
            if f == 10:
                await  self.build_immortal()
            if f == 11:
                await  self.build_obs()
            if f == 12:
                await  self.build_robo()
            if f == 13:
                await  self.build_stargate()
            if f == 14:
                await  self.build_void()
            if f == 15:
                await  self.scout()
            if f == 16:
                await self.retreat()
            if f == 17:
                await self.defend()
            if f == 18:
                await self.build_adept()
            if f == 19:
                await self.full_retreat()
            if f == 20:
                await self.attack_random_base()
            if f == 21:
                await self.clear_an_obstacle()
            if f == 22:
                await self.build_forge()
            if f == 23:
                await self.build_shield_battery()
            if f == 24:
                await self.build_cannon()
            if f == 25:
                await self.build_twilight()
            if f == 26:
                await self.build_dark_shrine()
            if f == 27:
                await self.build_dark_templar()
            if f == 28:
                await self.build_support_pylon()
            if f == 29:
                await self.attack_closest_building()

            self.past_moves.append({'game_state':np_t_game_state, 'f':f})

        except:
            traceback.print_exc()


    async def build_workers(self):
        if self.units(NEXUS).ready.noqueue and self.can_afford(PROBE) and len(self.workers) < 50:
            nexus = self.units(NEXUS).ready.noqueue.random
            await self.do(nexus.train(PROBE))


    async def build_pylons(self):

        if  self.can_afford(PYLON) and self.units(NEXUS).ready :
            # if len(self.owned_expansions) > 1:
            #     nexuses = [i for i in self.owned_expansions][1:]
            # else:
            #     nexuses  = [i for i in self.owned_expansions]
            # nexus = random.choice(nexuses)
            await self.build(PYLON, near=self.units(NEXUS).random.position.towards(self.game_info.map_center, random.randint(2, 15)))



    async def build_support_pylon(self):
        if self.can_afford(PYLON) and self.units(PYLON).ready:
            nexus = self.units(PYLON).ready.random
            await self.build(PYLON, near=nexus.position, max_distance=5)


    async def build_assimilators(self):
        for nexus in self.units(NEXUS).ready:
            vaspenes = self.state.vespene_geyser.closer_than(15.0, nexus)
            for vaspene in vaspenes:
                if not self.can_afford(ASSIMILATOR):
                    break
                worker = self.select_build_worker(vaspene.position)
                if worker is None:
                    break
                if not self.units(ASSIMILATOR).closer_than(1.0, vaspene).exists:
                    await self.do(worker.build(ASSIMILATOR, vaspene))


    async def expand(self):
        if self.can_afford(NEXUS):
            await self.expand_now()


    async def build_gateway(self):
        if self.units(PYLON).ready.exists and self.can_afford(GATEWAY):
            pylon = self.units(PYLON).ready.random
            await self.build(GATEWAY, near=pylon)


    async def build_stargate(self):
        if self.units(CYBERNETICSCORE).ready.noqueue.exists and self.can_afford(STARGATE):
            pylon = self.units(PYLON).ready.noqueue.random
            await self.build(STARGATE, near=pylon)


    async def build_robo(self):
        if self.units(CYBERNETICSCORE).ready.noqueue.exists and self.can_afford(ROBOTICSFACILITY):
            pylon = self.units(PYLON).ready.noqueue.random
            await self.build(ROBOTICSFACILITY, near=pylon)


    async def build_zealot(self):
        if self.units(GATEWAY).ready.noqueue and  self.can_afford(ZEALOT):
            g = random.choice(self.units(GATEWAY).ready.noqueue)
            await self.do(g.train(ZEALOT))


    async def build_stalker(self):
        if self.units(GATEWAY).ready.noqueue and self.can_afford(STALKER) and self.supply_left > 0 and self.units(CYBERNETICSCORE).ready:
            g = random.choice(self.units(GATEWAY).ready.noqueue)
            await self.do(g.train(STALKER))

    async def build_adept(self):
        if self.units(GATEWAY).ready.noqueue and self.can_afford(ADEPT) and self.supply_left > 0 and self.units(CYBERNETICSCORE).ready:
            g = random.choice(self.units(GATEWAY).ready.noqueue)
            await self.do(g.train(ADEPT))


    async def build_obs(self):
        if self.units(ROBOTICSFACILITY).ready.noqueue and self.can_afford(OBSERVER) and self.supply_left > 0:
            g = random.choice(self.units(ROBOTICSFACILITY).ready.noqueue)
            await self.do(g.train(OBSERVER))


    async def build_immortal(self):
        if self.units(ROBOTICSFACILITY).ready.noqueue and self.can_afford(IMMORTAL) and self.supply_left > 0:
            g = random.choice(self.units(ROBOTICSFACILITY).ready.noqueue)
            await self.do(g.train(IMMORTAL))


    async def build_void(self):
        if self.units(STARGATE).ready.noqueue and self.can_afford(VOIDRAY) and self.supply_left > 0:
            g = random.choice(self.units(STARGATE).ready.noqueue)
            await self.do(g.train(VOIDRAY))


    async def build_forge(self):
        if self.units(PYLON).ready.exists and self.can_afford(FORGE):
            pylon = self.units(PYLON).ready.random
            await self.build(FORGE, near=pylon)


    async def build_shield_battery(self):
        if self.units(PYLON).ready.exists and self.units(CYBERNETICSCORE).ready.exists and self.can_afford(SHIELDBATTERY):
            pylon = self.units(PYLON).ready.random
            await self.build(SHIELDBATTERY, near=pylon)


    async def build_cannon(self):
        if self.units(PYLON).ready.exists and self.units(FORGE).ready.exists and self.can_afford(PHOTONCANNON):
            pylon = self.units(PYLON).ready.random
            await self.build(PHOTONCANNON, near=pylon)


    async def build_twilight(self):
        if self.units(PYLON).ready.exists and self.units(CYBERNETICSCORE).ready.exists and self.can_afford(TWILIGHTCOUNCIL):
            pylon = self.units(PYLON).ready.random
            await self.build(TWILIGHTCOUNCIL, near=pylon)


    async def build_dark_shrine(self):
        if self.units(PYLON).ready.exists and self.units(TWILIGHTCOUNCIL).ready.exists and self.can_afford(DARKSHRINE):
            pylon = self.units(PYLON).ready.random
            await self.build(DARKSHRINE, near=pylon)


    async def build_dark_templar(self):
        if self.units(DARKSHRINE).ready.noqueue and self.units(GATEWAY).ready.noqueue and self.can_afford(IMMORTAL) and self.supply_left > 0:
            g = random.choice(self.units(GATEWAY).ready.noqueue)
            await self.do(g.train(DARKTEMPLAR))


    def random_location_variance(self, enemy_start_location):
        x = enemy_start_location[0]
        y = enemy_start_location[1]

        x += ((random.randrange(-20, 20))/100) * enemy_start_location[0]
        y += ((random.randrange(-20, 20))/100) * enemy_start_location[1]

        if x < 0:
            x = 0
        if y < 0:
            y = 0
        if x > self.game_info.map_size[0]:
            x = self.game_info.map_size[0]
        if y > self.game_info.map_size[1]:
            y = self.game_info.map_size[1]

        go_to = position.Point2(position.Pointlike((x,y)))
        return go_to


    async def scout(self):
        scout_sent = False
        if len(self.units(OBSERVER)) > 0:
            scout = self.units(OBSERVER)[0]
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                scout_sent = True
                await self.do(scout.move(move_to))

        if len(self.units(PROBE)) > 0 and not scout_sent:
            scout = random.choice(self.units(PROBE))
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                scout_sent = True
                await self.do(scout.move(move_to))


        ag_units = []
        for UNIT in aggressive_units:
            for s in self.units(UNIT).idle:
                ag_units.append(s)
        if ag_units and not scout_sent:
            scout = random.choice(ag_units)
            if scout.is_idle:
                enemy_location = self.enemy_start_locations[0]
                move_to = self.random_location_variance(enemy_location)
                await self.do(scout.move(move_to))


    async def build_cybernetic(self):
        if self.units(GATEWAY).ready.exists and self.can_afford(CYBERNETICSCORE):
            pylon = self.units(PYLON).ready.random
            await self.build(CYBERNETICSCORE, near=pylon)


    async def find_target(self, unit):
        pass
        # await asyncio.sleep(0.01)

        if len(self.known_enemy_units) > 0:

            return unit.closest_to(self.known_enemy_units)
        elif len(self.known_enemy_structures) > 0:
            return unit.closest_to(self.known_enemy_structures)
        else:
            return self.enemy_start_locations[0]


    async def defend(self):

        for UNIT in aggressive_units:
            for s in self.units(UNIT).idle:
                if len(self.known_enemy_units) > 0  and self.units(NEXUS).ready:
                    if not s.can_attack_air:
                        targets = [i for i in self.known_enemy_units if not i.is_flying]
                    else:
                        targets = [i for i in self.known_enemy_units]
                    target = get_closest(s, targets)
                    if target:
                        await self.do(s.attack(target))


    async def attack(self):
        for UNIT in aggressive_units:
            for s in self.units(UNIT):
                if not s.can_attack_air:
                    targets = [i for i in self.known_enemy_units if not i.is_flying]
                else:
                    targets = [i for i in self.known_enemy_units]
                target = get_closest(s, targets)
                if target:
                    await self.do(s.attack(target))


    async def attack_random_base(self):
        targets = self.known_enemy_structures(ORBITALCOMMAND)

        if targets:
            target = random.choice(targets)
            for UNIT in aggressive_units:
                for s in self.units(UNIT).idle:
                    if target:
                        await self.do(s.attack(target))


    async def attack_closest_building(self):
        targets = self.known_enemy_structures

        if targets:
            for UNIT in aggressive_units:
                for s in self.units(UNIT).idle:
                    target = get_closest(s, self.known_enemy_structures)
                    if target:
                        await self.do(s.attack(target))


    async def clear_an_obstacle(self):
        if self.state.destructables:
            target = random.choice(self.state.destructables)

            for UNIT in aggressive_units:
                for s in self.units(UNIT).idle:
                    if target:
                        await self.do(s.attack(target))


    async def retreat(self):
        for UNIT in aggressive_units:
            for s in self.units(UNIT):
                target = get_closest(s, self.units(NEXUS))
                if target:
                    await self.do(s.move(target.position))


    async def full_retreat(self):
        for UNIT in aggressive_units:
            for s in self.units(UNIT):
                await self.do(s.move(self.start_location))


    async def attack_own_building(self):

        if self.supply_used > 150:
            targets = [i for i in self.units if i.is_structure]
            target = random.choice(targets)

            for UNIT in aggressive_units:
                for s in self.units(UNIT).idle:
                    await self.do(s.attack(target))



    '''Helper functions'''
    def distance_nexus_to_enemy(self):
        return get_closest_distance(self.units(NEXUS), self.known_enemy_units)


    def distance_to_enemy(self):
        return get_closest_distance(self.units, self.known_enemy_units)


    def distance_to_enemy_buildings(self):
        return get_closest_distance(self.units, self.known_enemy_structures)


def run_games():
    s = Strat()

    ts = int(datetime.datetime.now().timestamp())
    a = None
    print('playing hard toss')
    a = run_game(maps.get("AbyssalReefLE"), [
        Bot(Race.Protoss, SentdeBot(s)),
        Computer(Race.Protoss, Difficulty.VeryEasy)
        ], realtime=False)

    print(a.result)
    if a.name == 'Defeat':
        dump_dict['score'] = 0
    else:
        dump_dict['score'] = 1

    with open('{0}/{1}_data.plk'.format(path, ts), 'wb') as f:
        pickle.dump(dump_dict, f)
    print('game_score:', dump_dict['score'])
    return dump_dict['score']


if __name__ == '__main__':
    games = 0
    wins = 0
    win_rate = 0
    difficulties = [0]
    record = []

    for i in range(20000):
        print('game num:', i)
        try:
            if i % 10 == 0 and i != 0:
                train_strat_model()
        except:
            traceback.print_exc()
        wins += run_games()
        games += 1

        win_rate = wins / games

        print('win_rate', win_rate, i)
        record.append({'win_rate':win_rate, 'num': i})
        df = pd.DataFrame.from_dict(record)
        df.to_csv('record.csv', index = False)


        # pool = [multiprocessing.Process(target=run_games) for i in range(6)]
        # for p in pool:
        #     p.start()
        #     time.sleep(5)
        # [p.join() for p in pool]


