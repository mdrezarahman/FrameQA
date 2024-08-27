# ------------------------------------------------------------------------------
# Chart tool
# ------------------------------------------------------------------------------
from typing import Dict, Any
from ..conversation.conversation import Conversation

#
# Declare the name of this tool and what it does
#
name = 'chart'

#
# Declare the invoke function
#


async def invokeTool(
        options: Dict[str, Any],
        *,
        metadata: Conversation.Metadata
) -> None:
    '''
    Output a chart
    '''
    Conversation.push(metadata, type='chart', value=options)
