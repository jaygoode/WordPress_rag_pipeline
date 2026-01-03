from .types import Message, PlanStep
from .controller import BaseAgentController

class AgentController(BaseAgentController):
    
    def __init__(self, retriever):
        self.retriever = retriever
        
    def plan(self, history):
        return [
            PlanStep(
                kind="retrieve",
                payload={"k": 5}
            ),
            PlanStep(
                kind="respond",
                payload={}
            )
        ]
    def run(self, history):
        user_query = history[-1].content
        chunks = self.retriever.search(user_query)
        answer = "\n\n".join(chunk.text for chunk in chunks)

        return Message(
            role="assistant",
            content=answer
        )


