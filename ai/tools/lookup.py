# ------------------------------------------------------------------------------
# Lookup  tool
# ------------------------------------------------------------------------------
from aparavi import debug
from ..conversation.conversation import Conversation

#
# Declare the name of this tool
#
name = 'lookup'

#
# Declare the invoke function
#


async def invokeTool(
        query: str,
        returnType: str,
        *,
        metadata: Conversation.Metadata
) -> str | None:
    '''
    Uses the distiller to pull data
    '''
    debug(f'        Query             : {query}')
    debug(f'        Format            : {returnType}')

    # Get the documents from search - needs to be rerouted to
    # back into the platform as an AQL statement

    docs = await metadata['search'].asearch(
        query=query,
        docFilter=metadata['docFilter']
    )

    # Output some debug info
    debug(f'        Found             : {len(docs)} documents')

    # Run the documents through our question interface to get the answer
    answer = await metadata['lookup'].alookup(
        query=query,
        returnType=returnType,
        docFilter=metadata['docFilter'],
        docs=docs
    )

    # Save the referenced documents, entities, follow up and processed docs
    Conversation.push(metadata,
                      docref=answer['documents'],
                      followUp=answer['followUp'],
                      processed=answer['processed'])

    # Output some debug info
    debug(f'        Answer            : {answer["answer"]}')
    debug(f'        Processed         : {answer["processed"]} documents')
    for doc in answer['documents']:
        debug(f'        Referenced        : {doc}')

    # Return the results - it will be a string or None
    return answer['answer']
