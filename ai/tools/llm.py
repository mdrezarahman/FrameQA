# ------------------------------------------------------------------------------
# Chart tool
# ------------------------------------------------------------------------------
from ..common.prompts import Prompts
from ..common.util import safeString
from ..conversation.conversation import Conversation

#
# Declare the name of this tool
#
name = 'llm'


async def invokeTool(
    instructions: str,
    context,
    *,
    metadata: Conversation.Metadata
) -> str:
    '''
    Use the LLM chat to carry out further processing
    '''

    # Stringify it
    contextItem = safeString(str(context))

    # Build the prompt
    prompt = Prompts.getPrompt('llm', {
        'instructions': instructions,
        'context': contextItem
    })

    # Call the llm
    answer = await metadata['chat'].achat(prompt)

    # Return it
    return answer
