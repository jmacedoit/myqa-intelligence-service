

import json
from collections import defaultdict

from services.embeddings_store import ResourceChunkInfo

def order_and_sew_info_chunks(info_chunks: list[ResourceChunkInfo]) -> list[ResourceChunkInfo]:
    # First we need to sort the chunks based on their chunk_number
    sorted_chunks = sorted(
        info_chunks, 
        key=lambda chunk: (chunk['resource_id'], json.loads(chunk['payload'])['chunk_number'])
    )

    sewed_chunks = []
    previous_chunk_number = -1
    previous_resource_id = None

    for i in range(len(sorted_chunks)):
        current_chunk_number = json.loads(sorted_chunks[i]['payload'])['chunk_number']
        current_resource_id = sorted_chunks[i]['resource_id']

        # If it's the first chunk or the chunk belongs to a different resource or 
        # the chunk_number is not one greater than the previous chunk_number, just append it to the list
        if i == 0 or previous_resource_id != current_resource_id or previous_chunk_number + 1 != current_chunk_number:
            sewed_chunks.append(sorted_chunks[i])
        else:
            # If not the first chunk of the resource and the chunk_number is one greater than 
            # the previous chunk_number, we need to find the overlap and merge the non-overlapping part
            previous_chunk_data = sorted_chunks[i - 1]['data']
            current_chunk_data = sorted_chunks[i]['data']
            overlap = find_overlap(previous_chunk_data, current_chunk_data)
            
            # Merge the non-overlapping part of the current chunk with the last chunk in the list
            sewed_chunks[-1]['data'] += current_chunk_data[len(overlap):]

        previous_chunk_number = current_chunk_number
        previous_resource_id = current_resource_id

    return sewed_chunks

def find_overlap(str1: str, str2: str) -> str:
    end_offset = min(len(str1), len(str2))
    for i in range(end_offset, 0, -1):
        if str1.endswith(str2[:i]):
            return str2[:i]
    return ""


def group_chunks_by_resource_id(chunks: list[ResourceChunkInfo]) -> dict[str, list[ResourceChunkInfo]]:
    grouped_chunks = defaultdict(list)
    for chunk in chunks:
        resource_id = chunk['resource_id']
        grouped_chunks[resource_id].append(chunk)

    return grouped_chunks
