# Y2k23
Solution for Polyglot challenge hosted by Hackerearth ([link](https://polyglot.hackerearth.com/))
Selected among the top 20 teams, total 830 teams participated in the hackathon.
Our approach:
The approach here is to convert event json to natural text, get the embeddings for this text and store it in a Weaviate vector database.
The problem is then posed as a Question and Answer (Q&A).
This can then be used to either create a conversational chatGPT kind of an interface, or to build a standardized event json which would work across different vendors as the queries are working at semantic level and not keyword based.
![architecture_polyglot](https://github.com/Shishuii/Y2k23/assets/22843318/a43d8b59-0241-4562-97c8-403bc48cd646)
