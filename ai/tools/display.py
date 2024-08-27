# ------------------------------------------------------------------------------
# Display tool
# ------------------------------------------------------------------------------
from typing import Any
from ..conversation.conversation import Conversation

#
# Declare the name of this tool
#
name = 'display'

#
# Declare the invoke function
#


async def invokeTool(
        item: Any,
        *,
        metadata: Conversation.Metadata
) -> None:
    '''
    Saves the input into given input into the final answer list
    '''
    Conversation.push(metadata, type='html', value=item)
