'''
OpenAI binding for the ChatLLM
'''
import asyncio
from ...common.chat import ChatBase
from ...common.config import Config
from langchain_openai import ChatOpenAI

class Chat(ChatBase):
    '''
    Creates an OpenAI chat bot
    '''

    '''
    Privates
    '''
    _model: str = ''
    _llm: ChatOpenAI

    def __init__(self):
        # Get our config
        config = Config.getConfig()

        # Pull out the section name
        sectionName = config['chat']

        # Get the section
        section = config[sectionName]

        # Get the API key
        key = section.get('genkey', "sk-xxxxxx")

        # Get the API server address
        openai_api_base = section.get('serverbase', None)

        # Get the model to use
        self._model = section.get('model', "gpt-3.5-turbo")

        # Init the chat base
        super().__init__()

        # Get the llm
        self._llm = ChatOpenAI(
            model=self._model,
            base_url=openai_api_base,
            api_key=key,
            temperature=0
        )

    async def achat(self, prompt: str) -> str:
        # debug(f'Prompt: {prompt}')

        # Get the plan
        results = await self._llm.ainvoke(prompt)

        # Get the response as a string
        content = str(results.content)

        # debug(f'Results: {content}')
        return content

    def chat(self, prompt: str) -> str:
        print("Run Chat")
        return asyncio.run(self.achat(prompt))

    def getTokens(self, value: str) -> int:
        '''
        This function will determine how many tokens, according to the model
        that the given string will take. This is used to prevent overflowing
        the prompt
        '''
        # Make sure we have a token counter
        return self._llm.get_num_tokens(value)
