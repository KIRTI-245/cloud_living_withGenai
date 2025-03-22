CREATE (t:Tenant {name: 'John Doe'})-[:RENTED]->(p:Property {location: 'Sector 45'})
CREATE (p)-[:HAS_REVIEW]->(r:Review {sentiment: 'Positive'})
