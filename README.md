# Y2k23
Solution for Polyglot challenge hosted by Hackerearth ([link](https://polyglot.hackerearth.com/))

Our approach:
The approach here is to convert event json to natural text, get the embeddings for this text and store it in a Weaviate vector database.
The problem is then posed as a Question and Answer (Q&A).
This can then be used to either create a conversational chatGPT kind of an interface, or to build a standardized event json which would work across different vendors as the queries are working at semantic level and not keyword based.
