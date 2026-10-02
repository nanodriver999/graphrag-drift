# Aura → Docker Neo4j logical dump

This step is intentionally separate from Leiden/community detection.

## Purpose

Copy the current Aura graph into a self-hosted Docker Neo4j instance without
requiring Aura snapshot API credentials.

This is a **logical graph dump**, not Neo4j's binary `.backup` / `.dump` format.

It preserves:

- node labels
- node properties
- relationship types
- relationship properties
- graph topology
- vector properties stored as numeric arrays

For source-to-target mapping, the importer adds `__source_element_id` to imported
nodes and relationships.

## Export Aura

```bash
python scripts/export_aura_logical_dump.py /tmp/aura-graph.json
```

Aura credentials are read from the existing `NEO4J_*` environment variables.

## Start Docker Neo4j

```bash
docker run -d --name graphrag-neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/local-test-password \
  neo4j:5.26-community
```

## Import

```bash
export LOCAL_NEO4J_URI=bolt://localhost:7687
export LOCAL_NEO4J_USERNAME=neo4j
export LOCAL_NEO4J_PASSWORD=local-test-password
export LOCAL_NEO4J_DATABASE=neo4j

python scripts/import_logical_dump.py /tmp/aura-graph.json
python scripts/verify_logical_dump.py /tmp/aura-graph.json
```

## Deliberate boundary

This step does **not** install GDS and does **not** run Leiden.

Leiden is implemented and tested in a separate follow-up branch/PR after this
data-copy pipeline is merged.
