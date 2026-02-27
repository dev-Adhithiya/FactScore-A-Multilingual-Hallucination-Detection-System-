import yaml, json
from core.retrieval import EmbeddingIndex

with open('config.yaml') as f:
    config = yaml.safe_load(f)

passages = []
with open('data/knowledge_base/passages.jsonl', encoding='utf-8') as f:
    for line in f:
        passages.append(json.loads(line))

print(f'Rebuilding index from {len(passages)} passages...')
index = EmbeddingIndex(config)
index.build_index(passages, save=True)
print('Done.')
