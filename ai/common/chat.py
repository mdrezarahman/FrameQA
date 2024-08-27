'''
This is the base class for the chat interface to the LLM
'''
import asyncio
import importlib
from abc import ABC, abstractmethod

from ..common.config import Config
from aparavi import debug


class ChatBase(ABC):
    '''
    This class is the base class for the interface into the Chat LLM
    '''

    '''
    Publics
    '''
    model = ''
    modelResponseTokens = 0
    modelTotalTokens = 0

    def __init__(self):
        '''
        Initialize the model info and token counters
        '''
        # Get our config
        config = Config.getConfig()

        # Pull out the section name
        section = config['chat']

        # Get the config variables
        self.model = config[section].get('model', "gpt-3.5-turbo")
        self.modelResponseTokens = config[section].get('modelResponseTokens', 4000)
        self.modelTotalTokens = config[section].get('modelTotalTokens', 4096)

        # Output some debug info
        debug('Chat:')
        debug(f'    Model             : {self.model}')
        debug(f'    Total tokens      : {self.modelTotalTokens}')
        debug(f'    Response tokens   : {self.modelResponseTokens}')

    @abstractmethod
    def getTokens(self, value: str) -> int:
        '''
        This function will determine how many tokens, according to the model
        that the given string will take. This is used to prevent overflowing
        the prompt
        '''

    @abstractmethod
    async def achat(self, prompt: str) -> str:
        '''
        Invoke the async chat interface. Return a string answer
        '''

    def chat(self, prompt: str) -> str:
        '''
        Invoke the chat interface. Return a string answer
        '''
        return asyncio.run(self.achat(
            prompt
        ))


def getChat() -> ChatBase:
    '''
    Looks at the configuration and returns and initializes a Chat
    '''
    # Get the configuration
    config = Config.getConfig()

    # Pull out the store we are going to use
    chat = config['chat']

    # Get the provider name to load
    provider = config[chat].get('provider', "openai")

    # Build up the module name - it will be in the store dir
    name = 'ai.providers.chat.' + provider

    # Get the module
    module = importlib.import_module(name)

    # Instantiate the chat class
    return getattr(module, 'Chat')()
