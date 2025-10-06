"""
Merge and deduplication aggregator for AffectRON.
Combines similar data points and removes duplicates.
"""

import json


logger = logging.getLogger(__name__)


class MergeDedupAggregator(BaseAggregator):
    """Aggregator that merges similar content and removes duplicates."""

    def __init__(self, config: AggregatorConfig, db_session):
        super().__init__(config, db_session)

    async def aggregate(self, data_batch: List[ExtractedData]) -> List[AggregatedResult]:
        """Aggregate data by merging similar content."""
        if not data_batch:
            return []

        # Group data by content similarity
        content_groups = self._group_similar_content(data_batch)

        merged_results = []

        for group_key, similar_items in content_groups.items():
            if len(similar_items) == 1:
                # No merging needed for single items
                continue

            # Merge similar items
            merged_data = self._merge_similar_items(similar_items)

            result = AggregatedResult(
                aggregator_name=self.config.name,
                aggregation_type="merge_dedup",
                data_points=[item.id for item in similar_items],
                result_data=merged_data,
                created_at=datetime.now()
            )

            merged_results.append(result)

        # Also create results for unique items that weren't merged
        processed_ids = set()
        for result in merged_results:
            processed_ids.update(result.data_points)

        unique_items = [item for item in data_batch if item.id not in processed_ids]

        for item in unique_items:
            result = AggregatedResult(
                aggregator_name=self.config.name,
                aggregation_type="unique_item",
                data_points=[item.id],
                result_data=self._extract_item_data(item),
                created_at=datetime.now()
            )
            merged_results.append(result)

        return merged_results

    def _group_similar_content(self, data_batch: List[ExtractedData]) -> Dict[str, List[ExtractedData]]:
        """Group data items by content similarity."""
        groups = defaultdict(list)

        for item in data_batch:
            # Generate content signature for grouping
            content_signature = self._generate_content_signature(item)
            groups[content_signature].append(item)

        # Filter out groups with only one item (no merging needed)
        return {k: v for k, v in groups.items() if len(v) > 1}

    def _generate_content_signature(self, item: ExtractedData) -> str:
        """Generate a signature for content grouping."""
        # Create a normalized signature based on key content elements
        title = item.title or ""
        content = item.content[:200] if item.content else ""  # First 200 chars

        # Extract key financial terms and numbers
        import re
        numbers = re.findall(r'\d+(?:[.,]\d+)?', f"{title} {content}")
        currencies = re.findall(r'\b(RON|EUR|USD|BTC|ETH)\b', f"{title} {content}", re.IGNORECASE)

        # Create signature
        signature_parts = [
            title.lower().strip(),
            content.lower().strip(),
            str(sorted(numbers)),
            str(sorted(currencies, key=str.upper))
        ]

        # Hash the signature for consistent grouping
        signature = "|".join(signature_parts)
        return hashlib.md5(signature.encode()).hexdigest()

    def _merge_similar_items(self, similar_items: List[ExtractedData]) -> Dict[str, Any]:
        """Merge multiple similar data items into one."""
        if len(similar_items) == 1:
            return self._extract_item_data(similar_items[0])

        # Extract data from all items
        titles = [item.title for item in similar_items if item.title]
        contents = [item.content for item in similar_items if item.content]
        urls = [item.url for item in similar_items if item.url]
        timestamps = [item.published_at for item in similar_items if item.published_at]

        # Merge titles (find most common or combine)
        merged_title = self._merge_titles(titles) if titles else None

        # Merge content (combine and deduplicate)
        merged_content = self._merge_content(contents)

        # Use earliest timestamp
        earliest_timestamp = min(timestamps) if timestamps else None

        # Combine URLs (remove duplicates)
        unique_urls = list(set(urls))

        # Extract metadata from all items
        all_metadata = []
        for item in similar_items:
            if item.metadata:
                try:
                    metadata = json.loads(item.metadata) if isinstance(item.metadata, str) else item.metadata
                    all_metadata.append(metadata)
                except:
                    continue

        merged_metadata = self._merge_metadata(all_metadata)

        return {
            'merged_title': merged_title,
            'merged_content': merged_content,
            'source_urls': unique_urls,
            'published_at': earliest_timestamp.isoformat() if earliest_timestamp else None,
            'original_item_count': len(similar_items),
            'metadata': merged_metadata,
            'merge_method': 'content_similarity'
        }

    def _merge_titles(self, titles: List[str]) -> str:
        """Merge multiple titles into one representative title."""
        if len(titles) == 1:
            return titles[0]

        # Find the most common title or create a combined one
        title_counts = defaultdict(int)
        for title in titles:
            title_counts[title] += 1

        most_common = max(title_counts.items(), key=lambda x: x[1])

        if most_common[1] > len(titles) / 2:
            # Majority agree on this title
            return most_common[0]
        else:
            # Combine titles
            return " | ".join(set(titles))[:200]  # Limit length

    def _merge_content(self, contents: List[str]) -> str:
        """Merge multiple content texts."""
        if len(contents) == 1:
            return contents[0]

        # Simple merging strategy: combine unique sentences
        all_sentences = set()
        for content in contents:
            # Split into sentences (basic implementation)
            sentences = [s.strip() for s in content.split('.') if s.strip()]
            all_sentences.update(sentences)

        # Join back into coherent text
        merged = '. '.join(sorted(all_sentences))[:1000]  # Limit length
        return merged

    def _merge_metadata(self, metadata_list: List[Dict]) -> Dict[str, Any]:
        """Merge metadata from multiple items."""
        if not metadata_list:
            return {}

        merged = {}

        # Combine source information
        sources = []
        for metadata in metadata_list:
            if 'source' in metadata:
                sources.append(metadata['source'])

        if sources:
            merged['sources'] = list(set(sources))

        # Combine entity information
        all_entities = []
        for metadata in metadata_list:
            if 'entities' in metadata:
                entities = metadata['entities']
                if isinstance(entities, str):
                    try:
                        entities = json.loads(entities)
                    except:
                        continue

                if isinstance(entities, dict):
                    for entity_type, entity_list in entities.items():
                        if isinstance(entity_list, list):
                            all_entities.extend(entity_list)

        if all_entities:
            # Deduplicate entities
            unique_entities = {}
            for entity in all_entities:
                if isinstance(entity, dict) and 'text' in entity:
                    key = entity['text'].lower()
                    if key not in unique_entities:
                        unique_entities[key] = entity

            merged['entities'] = list(unique_entities.values())

        return merged

    def _extract_item_data(self, item: ExtractedData) -> Dict[str, Any]:
        """Extract data from a single item."""
        return {
            'title': item.title,
            'content': item.content,
            'url': item.url,
            'published_at': item.published_at.isoformat() if item.published_at else None,
            'source_id': item.source_id,
            'metadata': item.metadata,
            'original_item_count': 1,
            'merge_method': 'none'
        }
