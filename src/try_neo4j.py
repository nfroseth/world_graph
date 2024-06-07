from neo4j import GraphDatabase, RoutingControl
from get_embedding import timing

URI = "neo4j://localhost:7687"
AUTH = ("neo4j", "neo4jneo4j")

# url = os.getenv("NEO4J_URI", "neo4j://localhost:7687")
# username = os.getenv("NEO4J_USER", "neo4j")
# password = os.getenv("NEO4J_PASSWORD", "neo4jneo4j")

def add_friend(driver, name, friend_name):
    driver.execute_query(
        "MERGE (a:Person {name: $name}) "
        "MERGE (friend:Person {name: $friend_name}) "
        "MERGE (a)-[:KNOWS]->(friend)",
        name=name, friend_name=friend_name, database_="neo4j",
    )

def print_friends(driver, name):
    records, _, _ = driver.execute_query(
        "MATCH (a:Person)-[:KNOWS]->(friend) WHERE a.name = $name "
        "RETURN friend.name ORDER BY friend.name",
        name=name, database_="neo4j", routing_=RoutingControl.READ,
    )
    for record in records:
        print(record["friend.name"])


@timing
def get_related_linked_note(driver, name):
    cypher_query = '''
        MATCH (p)-[r:RELATED_TO]->(n) WHERE p.name = $name
        RETURN n.name
        UNION
        MATCH (n)-[r:RELATED_TO]->(p) WHERE p.name = $name
        RETURN n.name
    '''
    # print(cypher_query)
    records, summary, keys = driver.execute_query(
        cypher_query, name=name, database_="neo4j", routing_=RoutingControl.READ
    )
    
    print(f"Cypher query {cypher_query}")
    for record in records:
        print(f'{record["n.name"]=}')

    return records

with GraphDatabase.driver(URI, auth=AUTH) as driver:
    # add_friend(driver, "Arthur", "Guinevere")
    # add_friend(driver, "Arthur", "Lancelot")
    # add_friend(driver, "Arthur", "Merlin")
    
    # search_note = "3-1-22 Two Months in with some progress"
    # search_note = "2-22-22 Frantic, yet exciting"
    search_note = "Typing"