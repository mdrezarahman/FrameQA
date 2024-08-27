'''
Question class
'''
import asyncio
from typing import List, TypedDict, cast, Any
from aparavi import debug
from ..search import Search
from ..common.prompts import Prompts
from ..common.distill import Distill, DistillResult
from ..common.chat import ChatBase, getChat
from ..common.schema import Doc, DocFilter, DocEntity, DocReference
from ..common.util import safeString, parseJson


class QuestionFinalResponse(TypedDict):
    answer: str
    entities: List[DocEntity]
    followUp: List[str]


class QuestionResult(TypedDict):
    answer: str
    entities: List[DocEntity]
    followUp: List[str]
    documents: List[DocReference]
    processed: int


class Question:
    '''
    The Question takes a series of documents and answers a question across many
    documents. It uses the underlying distill class.
    '''

    '''
    Privates
    '''
    _chat: ChatBase
    _distill: Distill

    def __init__(self):
        # Get  chat interface
        self._chat = getChat()

        # Bind the prompt
        self._distill = Distill(chat=self._chat)

    async def aquestion(
            self,
            query: str,
            *,
            docFilter: DocFilter,
            search: Search | None = None,
            docs: List[Doc] | None = None) -> QuestionResult:
        '''
        Given a query, a set of documents retrieved by the search
        endpoint, find the requested information
        '''
        # Setup our results
        result: QuestionResult = {
            'answer': '',
            'entities': [],
            'followUp': [],
            'documents': [],
            'processed': 0
        }

        # Declare the list of answers we are building
        answers: List[DistillResult] = []

        def parseResponse(content: str) -> QuestionFinalResponse:
            '''
            Clear up any missing fields and force into the ChatResponse type
            '''

            # Trim spaces
            content = content.strip()

            try:
                # Parse the json
                response = cast(QuestionFinalResponse, parseJson(content))

                # Make sure it has all the fields we need
                if 'answer' not in response:
                    print('            Question had no answer')
                    response['answer'] = ''
                if 'entities' not in response:
                    print('            Question had no entities')
                    response['entities'] = []
                if 'followUp' not in response:
                    print('            Question had no followUp')
                    response['followUp'] = []

            except BaseException:
                # If it is not json, use it directly as the result
                # Build the response with it directly - the llm messed up
                response = {
                    'answer': content,
                    'entities': [],
                    'followUp': []
                }

            # And return it
            return response

        async def askQuestion(query: str = '',
                              *,
                              docs: List[Doc] | None = None) -> DistillResult:
            # If we do not have docs, do a search
            if docs is None:
                # Search must be specified
                if search is None:
                    raise BaseException('Either docs or search must be specified')

                # Do the search
                docs = await search.asearch(query, docFilter=docFilter)

            # Get the initial answer
            answer = await self._distill.adistill(query,
                                                  docs=docs,
                                                  promptName='question-primary',
                                                  docFilter=docFilter,
                                                  returnType='str')
            return answer

        async def answerQuestion(question: str,
                                 *,
                                 docs: List[Doc] | None = None) -> DistillResult:
            nonlocal result, answers

            # Answer the question and return the results
            answer = await askQuestion(question, docs=docs)

            # If we have an anwer , add it
            if answer['answer'] is not None and answer['answer']:
                answers.append(answer)

            # Add to the total processed count
            result['processed'] += answer['processed']

            # Add the referenced documents
            for doc in answer['documents']:
                if all(doc.items() != d.items() for d in result['documents']):
                    result['documents'].append(doc)

            # Return the answer
            return answer

        async def answerQuestions(questions: List[str]) -> None:
            tasks: List[Any] = []

            # For each question, create an async task for it
            for question in questions:
                tasks.append(answerQuestion(question))

            # Wait until they are all done
            await asyncio.gather(*tasks)
            return

        # Do an initial question
        answer = await answerQuestion(query, docs=docs)

        # If there are no follow up questions or we have no search, just return this
        if not docFilter.followUpQuestions or search is None:
            # Fill in what we can - no entities
            result['answer'] = answer['answer']
            result['followUp'] = answer['followUp']
        else:
            # Now, get results for all the follow questions
            await answerQuestions(answer['followUp'])

            # Generate all the text
            text = ''
            for answer in answers:
                text = text + answer['answer'] + '\n\n'

            # Get an empty prompt so we can size it
            prompt = Prompts.getPrompt('question-final', {
                'query': query,
                'document': safeString(text)
            })

            # Call the llm
            content = await self._chat.achat(prompt)

            # Parse it
            finalAnswer = parseResponse(content)

            # Fill it in
            result['answer'] = finalAnswer['answer']
            result['entities'] = finalAnswer['entities']
            result['followUp'] = finalAnswer['followUp']

        # Print the result we got on this pass
        print(f'            Question result: {result["answer"]}')
        return result

    def question(
            self,
            query: str,
            *,
            docFilter: DocFilter,
            search: Search | None = None,
            docs: List[Doc] | None = None) -> QuestionResult:
        '''
        Given a query and a set of documents retrieved by the search
        endpoint, answer the question in a human readable form.
        '''
        return asyncio.run(self.aquestion(
            query=query,
            docFilter=docFilter,
            search=search,
            docs=docs
        ))
