"""Quick verification that cache fix worked."""
import json

cache = json.load(open('data/cache/spotify_search_cache_final_dataset_additional.json'))
print('Verifying fixed cache:')
print(f'Total entries: {len(cache)}')
print('\nChecking previously problematic entries:')
print('1. Childish Gambino:')
key1 = 'iii telegraph ave oakland by lloyd||childish gambino'
print(f'   Key: {key1}')
print(f'   Found: {key1 in cache}')
if key1 in cache:
    print(f'   Song: {cache[key1]["name"]} by {cache[key1]["artists"]}')

print('\n2. Macklemore:')
key2 = 'good old days feat kesha||macklemore'
print(f'   Key: {key2}')
print(f'   Found: {key2 in cache}')
if key2 in cache:
    print(f'   Song: {cache[key2]["name"]} by {cache[key2]["artists"]}')

print('\nCache fix verified!')

