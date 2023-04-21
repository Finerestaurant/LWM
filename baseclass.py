from abc import ABC, abstractmethod
from typing import Dict, Sequence, Optional, AnyStr

# Agent 자체에서 model의 parameter들을 self로 가지고 있지는 않으며,
# train을 포함한 backprop도 Agent내부에서 하지는 않음.
# Agent.train은 __init__에서 정의한 self.model의 backprop 함수를 실행시키는 방식으로 함.
# 즉, Agent.train은 torch의 치자면 모델의 loss.backward()를 단지 실행시키는 방식으로 작동.



class Agent(ABC):
    """
    Agent는 agent가 상속받아야 할 기본적인 method들을 정의합니다.    
    """

    @abstractmethod
    def generate_message(self, user_utterance: AnyStr) -> AnyStr:
        """user의 utterance를 받아서 agent의 utterance를 생성합니다.

        Argument:
        - user_utterance : 
        - 나머지 argument들은 model의 구조와 user의 구조에 따라서 추후 결정.
        """
        return None
    


class Env(ABC):
    """
    Env는 Env의 base 클래스를 정의합니다.

    observation space와 action space는 따로 지정하지 않습니다. 
    Env가 될 외부의 LM에서 받을수 있는 observation space와,
    Env와 상호작용할 Agent의 LM의 action space는 text로 동일하기 때문입니다. 

    dialogue history에 Env의 message와 agent의 message가 저장되어질때에, 
    Env와 agent는 모두의 말은 한꺼번에 observation하기 때문에, 
    language env에서의 observation space와 action space를 지정하는 것은 
    더더욱 그 의미가 퇴색되어 집니다. 
    """

    def step(self, action):
        raise NotImplementedError

    def close(self):
        pass
