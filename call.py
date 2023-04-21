from env import LangEnv
from baseclass import BaseAgent

env = LangEnv() # env 만들기.

agent = BaseAgent() # agent 만들기.

action = env.reset() # env.reset으로 첫 observation 가지고 오기. 
                     # 첫 observation은 langchain의 AIMessage.
                     # langchain의 Message를 생성하기 이전에 API key, system message, agent message를 가지고 와야함.  
                     # action space와 observation space가 어떻게 영향을 미치는지 알아보아야 할듯.
                     # 싶지만 서도, 그냥 Env class를 gym env말고 따로 만들어도 될듯 싶기도...
obs = env.step(action)