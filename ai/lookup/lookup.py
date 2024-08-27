# ------------------------------------------------------------------------------
# Question class
# ------------------------------------------------------------------------------
from typing import List

from ..common.distill import Distill, DistillResult
from ..common.chat import ChatBase, getChat
from ..common.schema import Doc, DocFilter

#
# Look up and distill values
#


class Lookup:
    '''
    The Lookup class takes a series of documents and extracts data across many
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

    async def alookup(
            self,
            query: str,
            docs: List[Doc],
            docFilter: DocFilter,
            returnType: str) -> DistillResult:
        '''
        Given a query and a set of documents retrieved by the search
        endpoint, extract the data
        '''
        return await self._distill.adistill(
            query=query,
            docs=docs,
            docFilter=docFilter,
            returnType=returnType,
            promptName='lookup'
        )
