from pydantic import BaseModel


class Transaction(BaseModel):

    # Transaction Features
    type: str
    amount: float
    oldbalanceOrg: float
    newbalanceOrig: float
    oldbalanceDest: float
    newbalanceDest: float

    # Graph Features (optional)
    sender_degree: float = 0.0
    sender_in_degree: float = 0.0
    sender_out_degree: float = 0.0
    sender_pagerank: float = 0.0
    sender_betweenness: float = 0.0
    sender_clustering: float = 0.0