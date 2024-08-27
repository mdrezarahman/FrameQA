# ------------------------------------------------------------------------------
# HTML tool
# ------------------------------------------------------------------------------
from ..conversation.conversation import Conversation

#
# Declare the name of this tool
#
name = 'html'

#
# Declare the invoke function
#


async def invokeTool(
        htmlCode: str,
        *,
        metadata: Conversation.Metadata
) -> None:
    # Display it
    Conversation.push(metadata, type='html', value=htmlCode)
