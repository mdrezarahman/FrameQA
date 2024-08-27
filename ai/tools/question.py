# ------------------------------------------------------------------------------
# Question tool
# ------------------------------------------------------------------------------
from aparavi import debug
from ..conversation.conversation import Conversation

#
# Declare the name of this tool
#
name = 'question'

#
# Declare the invoke function
#


async def invokeTool(
        question: str,
        *,
        metadata: Conversation.Metadata
) -> str | None:
    '''
    Uses the question engine to answer a question
    '''
    print(f'        Query             : {question}')

    # Get the documents from search - needs to be rerouted to
    # back into the platform as an AQL statement

    docs = await metadata['search'].asearch(
        query=question,
        docFilter=metadata['docFilter']
    )

    # Output some debug info
    print(f'        Found             : {len(docs)} documents')

    # Run the documents through our question interface to get the answer
    answer = await metadata['question'].aquestion(
        query=question,
        docFilter=metadata['docFilter'],
        docs=docs,
        search=metadata['search']
    )

    # Save the referenced documents, entities, follow up and processed docs
    Conversation.push(metadata,
                      docref=answer['documents'],
                      entities=answer['entities'],
                      followUp=answer['followUp'],
                      processed=answer['processed'])

    # Output some debug info
    print(f'        Answer            : {answer["answer"]}')
    print(f'        Processed         : {answer["processed"]} documents')
    for doc in answer['documents']:
        print(f'        Referenced        : {doc}')

    # Return the results - it will be a string or None
    return answer['answer']
