# ------------------------------------------------------------------------------
# Console test harnes
# ------------------------------------------------------------------------------
from typing import List, Optional
from ai.lookup import Lookup
from ai.search import Search
from ai.question import Question
from ai.conversation import Conversation
from ai.common.prompts import Prompts
from ai.common.schema import DocFilter
from colorama import Fore, Style

import torch

class Context:
    docFilter: DocFilter = DocFilter({
        "context": False,
        "content": True,
        "useTableConcat": False,
        "useGroupRank": False,
        "useQuickRank": False,
        "isTable": False
    })

    mode: str = 'conv'
    format: str = ''
    modemsg: str = ''
    history: List[str] = []

    search: Optional[Search] = None
    question: Optional[Question] = None
    conversation: Optional[Conversation] = None
    lookup: Optional[Lookup] = None


def printItem(type: str, items: List[str]):
    '''
    Print an item with different colors based on its array index
        [0] = black
        [1] = blue
        [2] = black
        etc
    '''
    # Start out in normal color with 20 columns used for the header
    color = False
    hdrlen = 20
    maxColumns = 120

    # We will always print the column
    col: int = hdrlen

    # Get the formatted header and formatted spaces
    header = f'    {type.ljust(hdrlen - 6, " ")}: '
    spaces = f'    {"".ljust(hdrlen -6, " ")}: '

    # Reset the style
    print(Style.RESET_ALL, end='')
    print(header, end='')

    # For each string specified
    for item in items:
        # Turn on highlighting
        if color:
            print(Fore.LIGHTBLUE_EX, end='')
        else:
            print(Style.RESET_ALL, end='')

        # Reverse it
        color = not color

        # Eliminate double \n - makes it look better
        item = item.replace('\n\n', '\n')

        # For each chr in the string
        for chr in item:
            # If we reached the end, put a column header out and reset column to beginning
            if col > maxColumns:
                print('\n', end='')
                print(spaces, end='')
                col = hdrlen

            # Output the chr
            if chr == '\n' or chr == '\r':
                print('\n', end='')
                print(spaces, end='')
                col = hdrlen
            else:
                print(chr, end='')
                col = col + 1

    # Reset
    print(Style.RESET_ALL)


def printItems(type: str, items: List[str]):
    '''
    Print an array of items
    '''
    for item in items:
        printItem(type, [item])
        type = ''


def statusUpdate(message: str) -> None:
    '''
    Update the status message callback
    '''
    print(Fore.LIGHTWHITE_EX, end='')
    print(message)
    print(Fore.RESET, end='')


def printMenu(context: Context):
    context.modemsg = 'unknown'

    print('Commands')
    print('-------------------------------------------------------')
    print(f'path    : {context.docFilter.parent}')
    print(f'limit   : {context.docFilter.limit}')
    print(f'format  : {context.format}')

    if context.docFilter.useQuickRank:
        print(
            'qrank   : ON           qrank                  toggle quick rerank')
    else:
        print(
            'qrank   : OFF          qrank                  toggle quick rerank')

    if context.docFilter.useGroupRank:
        print(
            'grank   : ON           grank                  toggle group rerank')
    else:
        print(
            'grank   : OFF          grank                  toggle group rerank')

    if context.docFilter.isTable:
        print(
            'tables  : ON           tables                 toggle table usage')
    else:
        print(
            'tables  : OFF          tables                 toggle table usage')

    if context.docFilter.useTableConcat:
        print(
            'tconcat : ON           tconcat                toggle table concatenation')
    else:
        print(
            'tconcat : OFF          tconcat                toggle table concatenation')

    if context.docFilter.context:
        print(
            'context : ON           context                toggle context return')
    else:
        print(
            'context : OFF          context                toggle context return')

    if context.docFilter.content:
        print(
            'content : ON           content                toggle content return')
    else:
        print(
            'content : OFF          content                toggle content return')

    if context.docFilter.isDeleted:
        print(
            'deleted : ON           deleted                toggle deleted document retrieval')
    else:
        print(
            'deleted : OFF          deleted                toggle deleted document retrieval')

    if context.docFilter.useSemanticSearch:
        print(
            'type    : SEMANTIC     keyword                set to use keyword search')
    else:
        print(
            'type    : KEYWORD      semantic               set to use semantic search')

    match context.mode:
        case 'question':
            context.modemsg = 'Question'
            print(
                'mode    : QUESTION     search,conv,lookup     set to change mode')

        case 'lookup':
            context.modemsg = 'Lookup'
            print(
                'mode    : LOOKUP       question,conv,search   set to change mode')

        case 'search':
            context.modemsg = 'Search'
            print(
                'mode    : SEARCH       question,conv,lookup   set to change mode')

        case _:
            context.modemsg = 'Conversation'
            print(
                'mode    : CONVERSATION search,question,lookup set to change mode')

            print(
                'reload  :                                     reload all text prompts')
            print(
                'clear   :                                     clear prior history')

    print('')


def search(context: Context, query: str) -> None:
    if context.search is None:
        context.search = Search()

    docs = context.search.search(
        query=query,
        docFilter=context.docFilter)

    print('')
    print('')
    print('')
    print('Results')
    print('')

    index = 0
    for doc in docs:
        print(
            f'Index             : {index}')
        print(
            f'Score             : {doc.score}')
        print(
            f'ObjectId          : {doc.metadata["objectId"]}')
        print(
            f'TableId:          : {doc.metadata["tableId"]}')
        print(
            f'Parent            : {doc.metadata["parent"]}')
        print(
            f'PermissionId      : {doc.metadata["permissionId"]}')
        print(
            f'Chunk:            : {doc.metadata["chunk"]}')
        print(
            '-----------------------------------------------------------------------')

        if context.docFilter.context and doc.page_content is not None:
            printItem('Context', [doc.page_content])

        if context.docFilter.content and doc.page_content is not None:
            printItem('Content', [doc.page_content])

        print('')
        print('')

        index = index + 1


def question(context: Context, query: str) -> None:
    if context.search is None:
        context.search = Search()
    if context.question is None:
        context.question = Question()

    answer = context.question.question(
        query=query,
        search=context.search,
        docFilter=context.docFilter)

    print('')
    print('')
    print('')
    print('Results')
    print(f'Question: {query}')
    print(
        '-----------------------------------------------------------------------')
    refDocs: List[str] = []
    for doc in answer['documents']:
        refDocs.append(doc['document'])

    entities: List[str] = []
    for entity in answer['entities']:
        entities.append(str(entity))

    printItems('Referenced', refDocs)
    printItems('Entities', entities)
    printItem('Answer', [answer['answer']])
    printItems('Follow-up', answer['followUp'])
    print('')
    print('')


def lookup(context: Context, query: str) -> None:
    if context.search is None:
        context.search = Search()
    if context.lookup is None:
        context.lookup = Lookup()

    # groups = context.search(
    #     query=query,
    #     docFilter=context.docFilter)

    # answer = context.lookup(
    #     query=query,
    #     returnType=context.format,
    #     groups=groups)

    # print('')
    # print('')
    # print('')
    # print('Results')
    # print(f'Question: {query}')
    # print(
    #     '-----------------------------------------------------------------------')
    # print(f'    Type          : {answer["type"]}')

    # docs = []
    # for doc in answer['documents']:
    #     docs.append(doc['document'])

    # printItems('Referenced', docs)
    # printItem('Answer', [str([answer['answer']])])
    # print('')
    # print('')


def conversation(context: Context, query: str) -> None:
    if context.conversation is None:
        context.conversation = Conversation()

    answer = context.conversation.conversation(
        query=query,
        docFilter=context.docFilter,
        history=context.history
    )

    context.history = answer['history']

    print('')
    print('')
    print('')
    print('Results')
    print(f'Conversation: {query}')
    print(
        '-----------------------------------------------------------------------')
    header = 'Context'
    if 'documents' in answer['documents']:
        for doc in answer['documents']:
            if not header:
                printItem('', ['--------------------'])

            printItem(header, [str(doc)])
            header = ''

    print('')
    print('')

    for value in answer['answers']:
        print(f'    Type          : {value["type"]}')

        if 'value' in value:
            printItem('Value', [str(value['value'])])
        print('')

    print('')
    print('')


def console() -> None:
    # Setup the context to pass around
    context = Context()

    # Output CUDA message
    if torch.cuda.is_available():
        print('GPU processing is enabled')
    else:
        print('GPU processing disabled. Recommend using GPU for better performance.')

    # Do forever
    while True:
        try:
            # Print the options menu
            printMenu(context)

            # Get a command
            query = input(f'{context.modemsg}> ')

            # Get the command part
            command = query.strip().split(' ', 1)[0].strip().lower()

            # Process the command
            match command:
                case 'reload':
                    Prompts.loadPrompts()

                case 'clear':
                    context.history = []

                case 'context':
                    context.docFilter.context = not context.docFilter.context

                case 'content':
                    context.docFilter.content = not context.docFilter.content

                case 'keyword':
                    context.docFilter.useSemanticSearch = False

                case 'semantic':
                    context.docFilter.useSemanticSearch = True

                case 'tables':
                    context.docFilter.isTable = not context.docFilter.isTable

                case 'tconcat':
                    context.docFilter.useTableConcat = not context.docFilter.useTableConcat

                case 'qrank':
                    context.docFilter.useQuickRank = not context.docFilter.useQuickRank

                case 'grank':
                    context.docFilter.useGroupRank = not context.docFilter.useGroupRank

                case 'question':
                    context.mode = 'question'

                case 'lookup':
                    context.mode = 'lookup'

                case 'search':
                    context.mode = 'search'

                case 'conv':
                    context.mode = 'conversation'

                case 'deleted':
                    context.docFilter.isDeleted = not context.docFilter.isDeleted

                case 'limit':
                    v = query.replace('  ', ' ').split(' ')
                    if len(v) != 2:
                        print(
                            "Command is 'limit n' where n is the number of documents to retrieve")
                    else:
                        context.docFilter.limit = int(v[1])

                case 'path':
                    v = query.replace('  ', ' ').split(' ', 1)
                    if len(v) != 2:
                        print(
                            "Command is 'path filterPath' where filter path is the prefix")
                    else:
                        context.docFilter.parent = v[1].strip().replace('\\', '/')

                case 'format':
                    v = query.replace('  ', ' ').split(' ', 1)
                    if len(v) != 2:
                        print(
                            "Command is 'format returnType' where returnType is the required return type")
                    else:
                        context.format = v[1].strip().replace('\\', '/')

                case _:
                    match context.mode:
                        case 'question':
                            question(context, query)
                        case 'lookup':
                            lookup(context, query)
                        case 'search':
                            search(context, query)
                        case _:
                            conversation(context, query)

            print('')

        except Exception as e:
            print()
            print(Fore.LIGHTRED_EX, end='')
            print(f'EXCEPTION: {e}')
            print(Style.RESET_ALL)
            print()
            pass


console()
