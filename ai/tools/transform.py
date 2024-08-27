# ------------------------------------------------------------------------------
# Chart tool
# ------------------------------------------------------------------------------
from typing import List, Any
import json
from ..conversation.conversation import Conversation
from ..common.prompts import Prompts

#
# Declare the name of this tool
#
name = 'transform'


async def invokeTool(
    instructions: str,
    values: List[Any],
    *,
    metadata: Conversation.Metadata
) -> Any:
    '''
    We must extract data
    '''

    # Add the items to the list of items to process removing any quotes so as not
    # to confuse the llm.
    items = []
    for value in values:
        if value is None:
            safe = 'None'
        else:
            safe = str(value).strip().replace('"', '\'')
        items.append(f'		 "{safe}",')

    # Join the items together
    itemStr = ''.join(items)

    # Build the prompt
    prompt = Prompts.getPrompt('transform', {
        'instructions': instructions,
        'items': itemStr
    })

    # Call the llm
    content = await metadata['chat'].achat(prompt)

    # Fix it in case the llm gave us a narative
    offset = content.find('```python')
    if offset >= 0:
        content = content[offset + 9:]
        offset = content.find('```')
        if offset >= 0:
            content = content[:offset]

    # Parse it and return it
    try:
        # Return the json after parsing
        return json.loads(content)

    except BaseException as e:
        Conversation.push(metadata, type='error', value=str(e))
        return None
