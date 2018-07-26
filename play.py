from model.player_first.player import Player
from game import Game

player_a=Player(1,0.8,"./model/player_first/model1/model.ckpt",step_type=0,step_top_n=5)
player_b=Player(-1,0.8,"./model/player_first/model2/model.ckpt",step_type=0,step_top_n=5)

game=Game("pos_model_2",player_a,player_b)

for i in range(200):
    game.start(1,0)
    print("no.{}".format(i))
