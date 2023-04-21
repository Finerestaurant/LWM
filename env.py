from typing import Any, Dict
from baseclass import Agent, Env
from langchain.chat_models import ChatOpenAI
from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)


class LangEnv(Env):

    def __init__(self, chatbot_api: ChatOpenAI, system_message, first_message):
        
        """
        우리의 Env가 되어줄 LM을 담아줄 class입니다. 

        Args:
            chatbot_api: chatbot을 명시해야합니다. 

            ex)
            from langchain.chat_models import ChatOpenAI

            chatbot = ChatOpenAI(config)
            env = LangEnv(chatbot)
        """
        super().__init__()

        self.first_message = first_message
        self.chatbot = chatbot_api # chatbot의 api
        self.env_message = None  
        self.system_message = system_message 
        self.prefix = 'Dialouge_history = '
        self.reset()

    def step(self, action: str):
        next_message = self.prefix + '[' + self.dial_history + '], ' + action

        self.dial_history = self.dial_history+ ', ' + action
        message = [
            SystemMessage(content=self.system_message),
            HumanMessage(content=next_message)
        ]

        self.env_message = self.chatbot(message).content
        self.dial_history = self.dial_history + ', ' + self.env_message

        return self.env_message

    def reset(self):
        """
        langchain의 api에게 system message와 human message를 넣어주어서 
        user simulator로써 self.env_message에 저장합니다. 
        """
        self.dial_history = self.first_message
        message = [
            SystemMessage(content=self.system_message),
            HumanMessage(content=self.first_message)
        ]
        # 첫 메세지.
        self.env_message = self.chatbot(message).content

        self.dial_history = self.dial_history + ', ' + self.env_message

