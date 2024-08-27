'''
Conversation class - implements the conversational interface
'''
import asyncio
import json
from ..common.chat import ChatBase, getChat
from ..common.prompts import Prompts
from ..common.util import parsePython
from ..common.schema import DocFilter, DocReference, DocEntity
from ..search import Search
from ..question import Question
from ..lookup import Lookup
from typing import List, Dict, Any, Callable, TypedDict, Optional
from functools import partial
from aparavi import debug

import glob
import importlib


class ConvTools(TypedDict):
    name: str
    func: Callable


class ConvAnswer(TypedDict):
    type: str
    value: Any


class ConvResult(TypedDict):
    answers: List[ConvAnswer]
    history: List[str]
    entities: List[DocEntity]
    followUp: List[str]
    documents: List[DocReference]
    processed: int


'''
    For every step you are about to take, call the status(msg: str) function to update the user
    on what you are going to do.
'''


class Conversation():
    '''
    Provides the main interface for the conversation
    '''

    class Metadata(TypedDict):
        self: Any
        chat: ChatBase
        search: Search
        question: Question
        lookup: Lookup
        docFilter: DocFilter
        history: List[str]
        results: ConvResult

    '''
    Static class variables. These can be overidden in the __init__
    if they are specified and attached to a specific instance
    '''
    _chat: ChatBase
    _search: Search
    _question: Question
    _lookup: Lookup
    _tools: List[ConvTools]

    @classmethod
    def push(cls,
             metadata: Metadata,
             *,
             type: str = '',
             value: Any = None,
             entities: List[DocEntity] = [],
             followUp: List[str] = [],
             docref: List[DocReference] = [],
             processed: int = 0):
        '''
        Pushes a value on the the answer stack...
        '''
        results = metadata['results']

        if type and value:
            # Create the answer
            answer: ConvAnswer = {
                'type': type,
                'value': value
            }

            # Save it on the answer list
            results['answers'].append(answer)

        # Add the referenced entities
        for entity in entities:
            if all(entity.items() != d.items() for d in results['entities']):
                results['entities'].append(entity)

        # Add the referenced documents
        for doc in docref:
            if all(doc.items() != d.items() for d in results['documents']):
                results['documents'].append(doc)

        # Add the suggest questions
        for suggest in followUp:
            if suggest not in results['followUp']:
                results['followUp'].append(suggest)

        # Add processed documents
        results['processed'] += processed

    def getStore(self):
        '''
        Returns the underlying store so it can be reused
        '''
        return self._search._store

    def getLookup(self):
        '''
        Returns the underlying lookup service so it can be reused
        '''
        return self._lookup

    def _loadTools(self) -> None:
        '''
        Loads all the tools from the tool directory
        '''
        # Debug
        debug('Tools:')

        # Reset it in case we load again
        Conversation._tools = []

        # Get the tools that are defined
        toolNames = glob.glob('*.py', root_dir='./ai/tools')
        for toolName in toolNames:
            # Build up the module name - it will be in the tools dir. Strip off
            # the .py at the end, which we know it has from the glob
            moduleName = 'ai.tools.' + toolName[:len(toolName) - 3]

            try:
                # Get the module
                module = importlib.import_module(moduleName)

                # Get the name, description and function
                name = getattr(module, 'name')
                func = getattr(module, 'invokeTool')

                # Create the tool and append it
                Conversation._tools.append({
                    'name': name,
                    'func': func
                })

                debug(f'    Tool              : {name} loaded')
            except BaseException as e:
                debug(f'    Tool              : {str(e)}')

    def __init__(self) -> None:
        # Save the drivers
        self._chat = getChat()
        self._search = Search()
        self._question = Question()
        self._lookup = Lookup()

        # Load the tools
        self._loadTools()

    async def aconversation(self,
                            query: str,
                            *,
                            history: Optional[List[str]] = None,
                            docFilter: Optional[DocFilter] = None,
                            statusCallback: Optional[Callable] = None) -> ConvResult:
        '''
        Runs the conversational chain
        '''
        def _restrictImports(module: str, *args, **kwargs):
            # Restrict the imports allowed to the following modules:
            importsAllowed = [
                'asyncio',
                'math',
                'pyproj'
            ]

            if module in importsAllowed:
                # Get the globals and built-ins we will be sharing
                builtins = globals()['__builtins__']

                # Call it
                return builtins['__import__'](module, *args, **kwargs)
            else:
                # Raise an error - not allowed
                raise BaseException(f'Importing module {module} not allowed')

        # Create a default filter if needed
        if docFilter is None:
            docFilter = DocFilter()

        # Create default history if needed
        if history is None:
            history = []

        # Setup our return
        results: ConvResult = {
            'answers': [],
            'history': [],
            'entities': [],
            'followUp': [],
            'documents': [],
            'processed': 0
        }

        # If there is nothing there
        query = query.strip()
        if not query:
            return results

        # Setup the metadata we pass around
        metadata: Conversation.Metadata = {
            'self': self,
            'chat': self._chat,
            'search': self._search,
            'question': self._question,
            'lookup': self._lookup,
            'docFilter': docFilter,
            'history': history,
            'results': results
        }

        # Get the globals and built-ins we will be sharing
        globs = globals()
        builtins = globals()['__builtins__']

        # Build up our locals
        evalLocals: Dict[str, Any] = {}

        # Build up our globals
        evalGlobals: Dict[str, Any] = {
            '__builtins__': {
                '__cached__': globs['__cached__'],
                '__file__': globs['__file__'],
                '__import__': _restrictImports,
                '__name__': globs['__name__'],
                '__package__': globs['__package__'],
                'all': builtins['all'],
                'bool': builtins['bool'],
                'complex': builtins['complex'],
                'dict': builtins['dict'],
                'enumerate': builtins['enumerate'],
                'Exception': builtins['Exception'],
                'False': builtins['False'],
                'float': builtins['float'],
                'hasattr': builtins['hasattr'],
                'int': builtins['int'],
                'isinstance': builtins['isinstance'],
                'len': builtins['len'],
                'list': builtins['list'],
                'map': builtins['map'],
                'max': builtins['max'],
                'min': builtins['min'],
                'object': builtins['object'],
                'print': builtins['print'],
                'property': builtins['property'],
                'range': builtins['range'],
                'reversed': builtins['reversed'],
                'round': builtins['round'],
                'set': builtins['set'],
                'slice': builtins['slice'],
                'str': builtins['str'],
                'sum': builtins['sum'],
                'True': builtins['True'],
                'type': builtins['type'],
                'zip': builtins['zip']
            }
        }

        # Build up the tool descriptions for the prompt and add the
        # invocation to the global
        toolDesc = ''
        for tool in Conversation._tools:
            # Grab the parameters from it
            _name = tool['name']
            _desc = Prompts.getPrompt('tools.' + tool['name'])
            _function = tool['func']
            _lambda = partial(_function, metadata=metadata)

            # Save the lambda in the local context
            evalGlobals[_name] = _lambda

            # Add it to the tool list
            toolDesc = toolDesc + '\n' + _desc

        # Debug
        debug(f'Conversation: {query}')

        # Get our base prompt - without history
        prompt = Prompts.getPrompt('conversation', {
            'history': '',
            'instructions': query,
            'tools': toolDesc
        })

        # Debug
        promptTokens = self._chat.getTokens(prompt)
        debug(f'    Base Input Tokens : {promptTokens}')

        # Now, build up the history
        historyList: List[str] = []
        historyTokens: List[int] = []

        if history is not None:
            # Walk through all items, and compute the token count
            for index in range(0, len(history)):
                # Get the history item
                historyItem = history[index]

                # Get the json string, add it to the list in reverse order and the tokens
                historyList.insert(0, historyItem)
                historyTokens.insert(0, self._chat.getTokens(historyItem))

            availTokens = self._chat.modelTotalTokens - promptTokens
            history = []
            for index in range(0, len(historyList)):
                # If this would overflow, stop
                if availTokens < historyTokens[index]:
                    break

                # Remove the tokens
                availTokens = availTokens - historyTokens[index]

                # Add to the beginning of the list - reversing it again
                history.insert(0, historyList[index])

        # Get our prompt with the conversational history now
        prompt = Prompts.getPrompt('conversation', {
            'history': ','.join(history),
            'instructions': query,
            'tools': toolDesc
        })

        # Debug
        promptTokens = self._chat.getTokens(prompt)
        debug(f'    Full Input Tokens : {promptTokens}')

        # Get the plan
        content = await self._chat.achat(prompt)

        # Parse it
        snippet = parsePython(content)

        # Setup the code with the snippet
        code = \
            'import asyncio\n' + \
            '\n' + \
            'async def exec_main():\n'

        lines = snippet.split('\n')
        for line in lines:
            code = code + '    ' + line + '\n'

        # Debug
        resultTokens = self._chat.getTokens(code)
        debug(f'    Result Tokens     : {resultTokens}')

        # Now, show the tasks
        debug('    Execution plan')
        debug(
            '    ------------------------------------------------------------')
        debug(code)
        debug(
            '    ------------------------------------------------------------')

        try:
            # Define the exec_main function
            exec(code, evalGlobals, evalLocals)

            # Now, execute it
            await evalLocals['exec_main']()
        except BaseException as e:
            # Save the error
            Conversation.push(metadata, type='error', value='I had a problem answering your question: ' + str(e))

        # Add this to the history
        history.append(json.dumps({
            "type": "instructions",
            "value": query
        }))

        # Add these results to our ongoing chat history
        for item in results['answers']:
            history.append(json.dumps({
                "type": "answer",
                "value": item
            }))

        # Return the history
        results['history'] = history
        print('---------------------------------------------')
        print('This is the prompt I gave you:')
        print(prompt)
        print()
        print('This was your answer:')
        print(code)
        print()
        print('I would like to ask you questions about your response')

        # Return it
        return results

    def conversation(self,
                     query: str,
                     *,
                     history: Optional[List[str]] = None,
                     docFilter: Optional[DocFilter] = None,
                     statusCallback: Optional[Callable] = None) -> ConvResult:
        '''
        Given a query and a set of documents retrieved by the search
        endpoint, answer the question in a human readable form.
        '''
        return asyncio.run(self.aconversation(
            query=query,
            history=history,
            docFilter=docFilter,
            statusCallback=statusCallback
        ))
