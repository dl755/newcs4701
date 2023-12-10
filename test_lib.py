from pypokerengine.players import BasePokerPlayer
from updated_game import setup_config, start_poker
import random
import plotly
import pickle
import pandas as pd
import os
import math


actions = {"preflop": [], "flop": [], "turn": [], "river": []}
action_dict = ["fold", "call", "raise"]

# Do not forget to make parent class as "BasePokerPlayer"

class refinedGTPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        if os.path.exists("update_payoffs.pkl"):
            update_payoffs = pickle.load(open("update_payoffs.pkl", "rb"))
        else:
            update_payoffs = {"preflop": [], "flop": [], "turn": [], "river": []}
        if os.path.exists("refined_tree.pkl"):
            tree = pickle.load(open("refined_tree.pkl", "rb"))
        else:
            tree = {"preflop": [], "flop": [], "turn": [], "river": []}
        ### PREFLOP
        if round_state['street'] == "preflop":
            preflop = tree['preflop']
            sorted_cards = hole_card.copy()
            sorted_cards.sort()
            if sorted_cards in [i[0] for i in preflop]:
                pos = [i[0] for i in preflop].index(sorted_cards)
                actions = preflop[pos]
                act_index = actions[1].index(max(actions[1]))
                action = action_dict[act_index]
                if action == "raise" and valid_actions[2]["amount"]["min"] > 0:
                    amount = valid_actions[2]["amount"]["min"]
                else:
                    amount = valid_actions[act_index]["amount"]
                update_payoffs['preflop'] = sorted_cards,action
                # print(update_payoffs)
                pickle.dump(update_payoffs, open("update_payoffs.pkl", "wb"))
                return action, amount
            else:
                tree['preflop'].append([sorted_cards, [0, 0, 0]])
                choice = random.randint(1,1)
                pickle.dump(tree, open("refined_tree.pkl", "wb"))
                update_payoffs['preflop'] = sorted_cards,action_dict[choice]
                pickle.dump(update_payoffs, open("update_payoffs.pkl", "wb"))
                return action_dict[choice], valid_actions[choice]["amount"]
            # if sorted_cards in [i[0] for i in flop]:
            #     pos = [i[0] for i in flop].index(sorted_cards)
            #     action_payoffs = flop[pos]
            #     act_index = action_payoffs[1].index(max(action_payoffs[1]))
            #     action = action_dict[act_index]
            #     if action == "raise" and valid_actions[2]["amount"]["min"] > 0:
            #         amount = valid_actions[2]["amount"]["min"]
            #     else:
            #         amount = valid_actions[act_index]["amount"]
            #     return action, amount
            # else:
            #     tree['flop'].append([sorted_cards, [0, 0, 0]])
            #     choice = random.randint(0,1)
            #     pickle.dump(tree, open("refined_tree.pkl", "wb"))
            #     return action_dict[choice], valid_actions[choice]["amount"]



            # if hole_card in [i[0] for i in preflop]:
            #     pos = [i[0] for i in preflop].index(hole_card)
            #     actions = preflop[pos]
            #     action, amount, payoff = "", 0, 0
            #     for i in actions:
            #         act = action_dict[i]
            #         amt = valid_actions[i]["amount"]
            #         if act == "raise":
            #             amt = valid_actions[i]["amount"]["min"]
            #         if i > payoff:
            #             if act == "raise":
            #                 if amt < 0:
            #                     act, amt = "call", valid_actions[1]["amount"]
            #             else:
            #                 act, amt = action_dict[i], valid_actions[i]["amount"]
            #                 payoff = i
            #     return act, amt
        ### FLOP
        elif round_state['street'] == "flop":
            flop = tree['flop']
            sorted_cards = round_state['community_card'].copy()
            sorted_cards.sort()
            if sorted_cards in [i[0] for i in flop]:
                pos = [i[0] for i in flop].index(sorted_cards)
                action_payoffs = flop[pos]
                act_index = action_payoffs[1].index(max(action_payoffs[1]))
                action = action_dict[act_index]
                if action == "raise" and valid_actions[2]["amount"]["min"] > 0:
                    amount = valid_actions[2]["amount"]["min"]
                else:
                    amount = valid_actions[act_index]["amount"]
                update_payoffs['flop'] = sorted_cards,action
                pickle.dump(update_payoffs, open("update_payoffs.pkl", "wb"))
                return action, amount
            else:
                tree['flop'].append([sorted_cards, [0, 0, 0]])
                choice = random.randint(0,1)
                pickle.dump(tree, open("refined_tree.pkl", "wb"))
                update_payoffs['flop'] = sorted_cards,action_dict[choice]
                pickle.dump(update_payoffs, open("update_payoffs.pkl", "wb"))
                return action_dict[choice], valid_actions[choice]["amount"]
        elif round_state['street'] == "turn":
            turn = tree['turn']
            if round_state['community_card'][-1] in [i[0] for i in turn]:
                pos = [i[0] for i in turn].index(round_state['community_card'][-1])
                action_payoffs = turn[pos]
                act_index = action_payoffs[1].index(max(action_payoffs[1]))
                action = action_dict[act_index]
                if action == "raise" and valid_actions[2]["amount"]["min"] > 0:
                    amount = valid_actions[2]["amount"]["min"]
                else:
                    amount = valid_actions[act_index]["amount"]
                update_payoffs['turn'] = round_state['community_card'][-1], action
                pickle.dump(update_payoffs, open("update_payoffs.pkl", "wb"))
                return action, amount
            else:
                tree['turn'].append([round_state['community_card'][-1], [0, 0, 0]])
                choice = random.randint(0,1)
                pickle.dump(tree, open("refined_tree.pkl", "wb"))
                update_payoffs['turn'] = round_state['community_card'][-1], action_dict[choice]
                pickle.dump(update_payoffs, open("update_payoffs.pkl", "wb"))
                return action_dict[choice], valid_actions[choice]["amount"]
                
        else:
            river = tree['river']
            if round_state['community_card'][-1] in [i[0] for i in river]:
                pos = [i[0] for i in river].index(round_state['community_card'][-1])
                action_payoffs = river[pos]
                act_index = action_payoffs[1].index(max(action_payoffs[1]))
                action = action_dict[act_index]
                if action == "raise" and valid_actions[2]["amount"]["min"] > 0:
                    amount = valid_actions[2]["amount"]["min"]
                else:
                    amount = valid_actions[act_index]["amount"]
                update_payoffs['river'] = round_state['community_card'][-1], action
                pickle.dump(update_payoffs, open("update_payoffs.pkl", "wb"))
                return action, amount
            else:
                tree['river'].append([round_state['community_card'][-1], [0, 0, 0]])
                choice = random.randint(0,1)
                pickle.dump(tree, open("refined_tree.pkl", "wb"))
                update_payoffs['river'] = round_state['community_card'][-1], action_dict[choice]
                pickle.dump(update_payoffs, open("update_payoffs.pkl", "wb"))
                return action_dict[choice], valid_actions[choice]["amount"]
                        


    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


class UpdatedRandomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        if round_state['street'] == "preflop":
            if len(actions["preflop"]) == 0:
                action, amount = random.choice(valid_actions)
                if action == "raise":
                    amount = amount["min"]
                return action, amount
            elif len(actions["preflop"]) == 1:
                action, amount = random.choice(valid_actions)
                if action == "raise":
                    return "call", 0
                return action, amount
        elif round_state['street'] == "flop":
            if len(actions["flop"]) == 0:
                action, amount = random.choice(valid_actions)
                if action == "raise":
                    amount = amount["min"]
                return action, amount
            elif len(actions["flop"]) == 1:
                action, amount = random.choice(valid_actions)
                if action == "raise":
                    return "call", 0
                return action, amount
        elif round_state['street'] == "turn":
            if len(actions["turn"]) == 0:
                action, amount = random.choice(valid_actions)
                if action == "raise":
                    amount = amount["min"]
                return action, amount
            elif len(actions["turn"]) == 1:
                action, amount = random.choice(valid_actions)
                if action == "raise":
                    return "call", 0
                return action, amount
        else:
            if len(actions["river"]) == 0:
                action, amount = random.choice(valid_actions)
                if action == "raise":
                    amount = amount["min"]
                return action, amount
            elif len(actions["river"]) == 1:
                action, amount = random.choice(valid_actions)
                if action == "raise":
                    return "call", 0
                return action, amount
        return action, amount   # action returned here is sent to the poker engine
    
    
    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


class FishPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [fold_action_info, call_action_info, raise_action_info]
        call_action_info = valid_actions[1]
        call_action, call_amount = call_action_info["action"], call_action_info["amount"]
        fold_action_info = valid_actions[0]
        fold_action = fold_action_info["action"]
        fold_amount = fold_action_info["amount"]
        containsGoodCard = False
        for i in hole_card:
            if "J" in i or "Q" in i or "K" in i or "A" in i:
                containsGoodCard = True
        percent_call = random.randint(1, 100)
        if containsGoodCard:
            return call_action, call_amount
        else:
            if percent_call < 90:
                return call_action, call_amount
            else:
                return fold_action, fold_amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


class RandomPlayer(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        raise_action_info = valid_actions[2]
        raise_action = raise_action_info["action"]
        raise_amount = raise_action_info['amount']['min']
        call_action_info = valid_actions[1]
        call_action, call_amount = call_action_info["action"], call_action_info["amount"]
        fold_action = valid_actions[0]["action"]
        fold_amount = valid_actions[0]["amount"]
        pairs_of_actions = [(raise_action, raise_amount),
                            (call_action, call_amount), (fold_action, fold_amount)]
        if raise_amount < 0:
            pairs_of_actions = [(call_action, call_amount),
                                (fold_action, fold_amount)]
        action, amount = random.choice(pairs_of_actions)
        return action, amount   # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


class AggressiveNLHAgent(BasePokerPlayer):
    def declare_action(self, valid_actions, hole_card, round_state):
        # valid_actions format => [raise_action_info, call_action_info, fold_action_info]
        raise_action_info = valid_actions[2]
        raise_action = raise_action_info["action"]
        print(raise_action_info['amount'])
        min_raise_amt = raise_action_info['amount']['min']
        max_raise_amt = raise_action_info['amount']['max']
        raise_amount = random.randint(
            min_raise_amt, min_raise_amt + (max_raise_amt - min_raise_amt) // 10)
        call_action_info = valid_actions[1]
        call_action, call_amount = call_action_info["action"], call_action_info["amount"]
        fold_action = valid_actions[0]["action"]
        fold_amount = valid_actions[0]["amount"]
        pairs_of_actions = [(raise_action, raise_amount),
                            (call_action, call_amount), (fold_action, fold_amount)]
        action, amount = random.choice(pairs_of_actions)
        if raise_amount < 0:
            return call_action, call_amount
        # action returned here is sent to the poker engine
        return raise_action, raise_amount

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


class FlopNode():
    def __init__(self, cards):
        self.cards = cards
        self.actions = []
        self.street = "preflop"
        self.payoff = 0

    def getCards(self):
        return self.cards


class ActionNode():
    def __init__(self, action, amount):
        self.action = action
        self.amount = amount
        self.payoff = 0


class GameTreePlayer(BasePokerPlayer):
    def getTree(self):
        if os.path.exists("tree.pkl"):
            tree = pickle.load(open("tree.pkl", "rb"))
        else:
            tree = {"node_type": "player", "cards": [],
                    "street": "preflop", "payoff": 0}

        return tree

    def preflop(self, valid_actions, hole_card, round_state):
        if round_state["street"] == "preflop":
            start = self.getTree()
            if hole_card in [i['cards'] for i in start["cards"]]:
                for i in start["cards"]:
                    if i["cards"] == hole_card:
                        curr = i
                        break
                if len(curr["actions"]) == 0:
                    action, amount = random.choice(valid_actions)
                    if action == "raise":
                        amount = random.randint(valid_actions[2]["amount"]["min"], (
                            valid_actions[2]["amount"]["max"]-valid_actions[2]["amount"]["min"])//10)
                    actionNode = {"node_type": "action", "action": action,
                                  "amount": amount, "payoff": 0}
                    curr["actions"].append(actionNode)
                else:
                    max_action, max_amount, max_payoff = "", 0, 0
                    action_amt_payoff_tups = []
                    for action in valid_actions:
                        if action in [i["action"] for i in curr["actions"]]:
                            for i in curr["actions"]:
                                if i["action"] == action:
                                    action_amt_payoff_tups.append(
                                        (i["action"], i["amount"], i["payoff"]))
                        else:
                            if action == "raise":
                                action_amt_payoff_tups.append(
                                    ("raise", 5, -10))
                            else:
                                if action == "call":
                                    payoff = -5
                                elif action == "fold":
                                    payoff = -1
                                else:
                                    payoff = 0
                                action_amt_payoff_tups.append(
                                    (action["action"], action["amount"], payoff))
                    for action, amount, payoff in action_amt_payoff_tups:
                        if payoff > max_payoff:
                            max_action, max_amount, max_payoff = action, amount, payoff
                        actionNode = {"node_type": "action", "action": action,
                                      "amount": amount, "payoff": payoff}
                        curr["actions"].append(actionNode)
                if max_action not in valid_actions:
                    return "call", 0

                return max_action, max_amount
            else:
                # create a new node for the hole card
                newNode = {"node_type": "flop", "cards": [], "actions": [],
                           "street": "preflop", "payoff": 0}
                action, amount = random.choice(valid_actions)
                if action == "raise":
                    amount = random.randint(valid_actions[2]["amount"]["min"], (
                        valid_actions[2]["amount"]["max"]-valid_actions[2]["amount"]["min"])//10)
                actionNode = {"node_type": "action", "action": action,
                              "amount": amount, "payoff": 0}
                newNode["actions"].append(actionNode)
                newNode["cards"] = hole_card
                start["cards"].append(newNode)
            pickle.dump(start, open("tree.pkl", "wb"))

    def flop(self, valid_actions, hole_card, round_state):
        assert round_state["street"] == "flop"
        start = {"node_type": "player", "flop": []}
        flopNode = {"node_type": "flop",
                    "cards": round_state["community_card"], "actions": [], }
        if flopNode['cards'] not in [i['cards'] for i in start["flop"]]:
            start["flop"].append(flopNode)
        pass

    def turn(self, valid_actions, hole_card, round_state):
        pass

    def river(self, valid_actions, hole_card, round_state):
        pass

    def declare_action(self, valid_actions, hole_card, round_state):
        if round_state["street"] == "preflop":
            self.preflop(valid_actions, hole_card, round_state)
        elif round_state["street"] == "flop":
            self.flop(valid_actions, hole_card, round_state)
        elif round_state["street"] == "turn":
            self.turn(valid_actions, hole_card, round_state)
        else:
            self.river(valid_actions, hole_card, round_state)
        start = self.getTree()
        fold_action_info = valid_actions[2]
        return "raise", 5   # action returned here is sent to the poker engine

    def receive_game_start_message(self, game_info):
        pass

    def receive_round_start_message(self, round_count, hole_card, seats):
        pass

    def receive_street_start_message(self, street, round_state):
        pass

    def receive_game_update_message(self, action, round_state):
        pass

    def receive_round_result_message(self, winners, hand_info, round_state):
        pass


class CFRPlauer(BasePokerPlayer):
    pass


def eval():
    avg_stack = [0, 0, 0]
    iters = 100
    for _ in range(0, iters):
        config = setup_config(max_round=10, initial_stack=200,
                              small_blind_amount=1)
        config.register_player(name="GameTree", algorithm=GameTreePlayer())
        config.register_player(name="Fish", algorithm=FishPlayer())
        config.register_player(
            name="Aggressive", algorithm=AggressiveNLHAgent())
        game_result = start_poker(config, verbose=1)
        for j in range(0, 3):
            # RANDOM FISH AGGRESSIVE
            avg_stack[j] += game_result["players"][j]["stack"]
    avg_stack = [x / iters for x in avg_stack]
    if os.path.exists("eval_agents.pkl"):
        avg_stack_df = pickle.load(open("eval_agents.pkl", "rb"))
        new_df = pd.DataFrame(columns=["Random", "Fish", "Aggressive"])
        new_df.loc[0] = avg_stack
        avg_stack_df = pd.concat([avg_stack_df, new_df], ignore_index=True)
    else:
        avg_stack_data = [{
            "Random": avg_stack[0], "Fish": avg_stack[1], "Aggressive": avg_stack[2]}]
        avg_stack_df = pd.DataFrame(columns=["Random", "Fish", "Aggressive"])
        avg_stack_df.loc[0] = avg_stack
    pickle.dump(avg_stack_df, open("eval_agents.pkl", "wb"))
    print(avg_stack_df)

def updatePayoffs(game_result, name="GameTree"):
    # print("trying to find stack")
    # print(game_result["players"])
    first = game_result["players"][0]["name"]
    if first == name:
        update_payoff_score = game_result["players"][0]["stack"] - 200
    else:
        update_payoff_score = game_result["players"][1]["stack"] - 200
    a = pickle.load(open("update_payoffs.pkl", "rb"))
    update_dict = {"fold": 0, "call": 1, "raise": 2}
    # print("this is what we need to update")
    # print(a)
    tree = pickle.load(open("refined_tree.pkl", "rb"))
    for i in a.keys():
        if a[i] != []:
            card, action = a[i]
            # print([j[0] for j in tree[i]])
            find_card = [j[0] for j in tree[i]].index(card)
            tree[i][find_card][1][update_dict[action]] += update_payoff_score/10
    # print(tree)
    pickle.dump(tree, open("refined_tree.pkl", "wb"))
            

def eval_game_tree():
    avg_stack = [0, 0]
    iters = 100
    for _ in range(0, iters):
        config = setup_config(max_round=10, initial_stack=200,
                              small_blind_amount=1)
        config.register_player(name="GameTree", algorithm=refinedGTPlayer())
        config.register_player(name="Random", algorithm=RandomPlayer())
        game_result = start_poker(config, verbose=1)
        for j in range(0, 2):
            # AI BOT RANDOM
            avg_stack[j] += game_result["players"][j]["stack"]
    avg_stack = [x / iters for x in avg_stack]
    if os.path.exists("eval_ai.pkl"):
        avg_stack_df = pickle.load(open("eval_ai.pkl", "rb"))
        new_df = pd.DataFrame(columns=["Custom Agent", "Random"])
        new_df.loc[0] = avg_stack
        avg_stack_df = pd.concat([avg_stack_df, new_df], ignore_index=True)
    else:
        avg_stack_df = pd.DataFrame(columns=["Custom Agent", "Random"])
        avg_stack_df.loc[0] = avg_stack
    pickle.dump(avg_stack_df, open("eval_ai.pkl", "wb"))
    print(avg_stack_df)



def gameTree():
    config = setup_config(max_round=1, initial_stack=200,
                          small_blind_amount=10)
    config.register_player(name="GameTree", algorithm=refinedGTPlayer())
    config.register_player(name="Fish", algorithm=FishPlayer())
    pickle.dump(actions, open("actions.pkl", "wb"))
    game_result = start_poker(config, verbose=0)
    # print(game_result)
    updatePayoffs(game_result)
    # updatePayoffs(game_result, name="GameTreeCopy")
    # print(game_result)
    # print(config.action_histories)


if __name__ == "__main__":
    # config = setup_config(max_round=10, initial_stack=200,
    #                       small_blind_amount=1)
    # config.register_player(name="Random", algorithm=RandomPlayer())
    # config.register_player(name="Fish", algorithm=FishPlayer())
    # config.register_player(name="Aggressive", algorithm=AggressiveNLHAgent())
    # game_result = start_poker(config, verbose=1)
    # eval()
    for i in range(0, 10):
        eval_game_tree()
