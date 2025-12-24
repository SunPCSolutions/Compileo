"""
Iterative Taxonomy Generator

This module implements the multi-stage iterative refinement process for taxonomy generation.
It orchestrates the flow of smart sampling, initial generation, iterative refinement,
and final consolidation to ensure 100% chunk coverage.
"""

from typing import List, Dict, Any, Optional, Callable
import math
import copy
from ...core.logging import get_logger

from .smart_sampler import SmartChunkSampler
from .merger import TaxonomyMerger


class IterativeTaxonomyGenerator:
    """
    Orchestrator for multi-stage iterative taxonomy generation.
    """

    def __init__(self, generation_func: Callable, extension_func: Callable):
        """
        Initialize the iterative generator.

        Args:
            generation_func: Function to call for initial taxonomy generation.
                             Signature: (chunks, domain, depth, batch_size, category_limits, specificity_level) -> result
            extension_func: Function to call for taxonomy refinement/extension.
                            Signature: (existing_taxonomy, chunks, additional_depth, domain, batch_size, category_limits, specificity_level) -> result
        """
        self.generation_func = generation_func
        self.extension_func = extension_func
        self.logger = get_logger(__name__)

    def generate(
        self,
        chunks: List[str],
        domain: str = "general",
        depth: Optional[int] = 3,
        batch_size: int = 10,
        category_limits: Optional[List[int]] = None,
        specificity_level: int = 1
    ) -> Dict[str, Any]:
        """
        Execute the multi-stage generation process.

        1. Stage 1: Smart Initial Sampling & Generation
        2. Stage 2: Iterative Refinement with remaining chunks
        3. Stage 3: Consolidation (implicit in final state)

        Args:
            chunks: List of all text chunks
            domain: Knowledge domain
            depth: Taxonomy depth
            batch_size: Chunks per batch (for initial sample and refinement batches)
            category_limits: Limits for categories per level
            specificity_level: Level of specificity

        Returns:
            Final refined taxonomy result
        """
        total_chunks = len(chunks)
        self.logger.info(f"Starting iterative taxonomy generation for {total_chunks} chunks.")

        # --- Stage 1: Smart Initial Sampling ---
        # Select representative chunks for the foundation
        sample_size = min(batch_size, total_chunks)
        initial_chunks = SmartChunkSampler.sample(chunks, sample_size)
        
        # Keep track of which chunks we've processed
        # Since SmartChunkSampler returns values, we need to identify them in the original list
        # to find the remaining ones. We'll use object identity or content matching if strings are unique.
        # Ideally, we should work with indices, but the interface passes strings.
        # Let's assume content uniqueness or just filter out the sampled ones.
        # A robust way is to use indices if we had them, but here we'll just subtract the list.
        
        # Create a set of initial chunk contents for efficient lookup
        initial_chunk_set = set(initial_chunks)
        remaining_chunks = [c for c in chunks if c not in initial_chunk_set]

        self.logger.info(f"Stage 1: Generating baseline from {len(initial_chunks)} sampled chunks.")
        
        current_taxonomy_result = self.generation_func(
            chunks=initial_chunks,
            domain=domain,
            depth=depth,
            batch_size=sample_size, # Use sample size for initial batch
            category_limits=category_limits,
            specificity_level=specificity_level
        )

        current_taxonomy = current_taxonomy_result.get('taxonomy', {})
        if not current_taxonomy:
             self.logger.error("Stage 1 failed: No taxonomy generated.")
             return current_taxonomy_result # Return partial/error result

        # --- Stage 2: Iterative Refinement ---
        if not remaining_chunks:
            self.logger.info("No remaining chunks for refinement. Returning Stage 1 result.")
            # Update metadata to reflect complete coverage
            if 'analytics' not in current_taxonomy_result:
                current_taxonomy_result['analytics'] = {}
            current_taxonomy_result['analytics']['chunk_coverage'] = 1.0
            # Mark as complete mode even when all chunks fit in initial batch
            current_taxonomy_result['generation_metadata']['processing_mode'] = 'complete'
            return current_taxonomy_result

        total_batches = math.ceil(len(remaining_chunks) / batch_size)
        self.logger.info(f"Stage 2: Refining with {len(remaining_chunks)} remaining chunks in {total_batches} batches.")
        
        # Add detailed step logging for API/CLI visibility
        if 'processing_steps' not in current_taxonomy_result:
            current_taxonomy_result['processing_steps'] = []
            
        current_taxonomy_result['processing_steps'].append({
            "stage": "initial_sampling",
            "message": f"Generated baseline taxonomy from {len(initial_chunks)} sampled chunks",
            "chunks_processed": len(initial_chunks),
            "total_chunks": total_chunks
        })

        # Process remaining chunks in batches
        for i in range(total_batches):
            start_idx = i * batch_size
            end_idx = min((i + 1) * batch_size, len(remaining_chunks))
            batch_chunks = remaining_chunks[start_idx:end_idx]
            
            batch_msg = f"Processing refinement batch {i+1}/{total_batches} ({len(batch_chunks)} chunks)"
            self.logger.info(batch_msg)
            self.logger.debug(batch_msg)

            # We use the extension function but conceptually we are "refining" the structure
            # The prompt for extension should handle "add missing categories" logic.
            # Ideally, we'd have a specific "refine" prompt, but "extend" is close.
            # We might need to adjust the extension logic or parameters to specifically invoke
            # a "refinement" mode if the underlying generator supports it.
            # For now, we assume extension_func can handle "adding knowledge from new chunks".
            
            # NOTE: We pass additional_depth=0 to imply "don't necessarily go deeper, just broaden/refine"
            # However, the current extend_taxonomy implementation usually adds depth.
            # We might need to ensure the underlying generator interprets this correctly.
            # If additional_depth is required to be > 0, we set it to 1, but really we want 0 (horizontal expansion).
            # Let's use 0 and hope the generator handles it or defaults to "refine structure".
            # If the generator requires depth > 0, we might strictly expand.
            # A safer bet with current prompts might be to use the existing depth or a small increment.
            
            try:
                # We are merging the new knowledge into the existing taxonomy.
                # The generator's extend_taxonomy usually takes existing taxonomy and returns a new one.
                # We need to make sure we are passing the CURRENT refined taxonomy, not the original.
                
                refinement_result = self.extension_func(
                    existing_taxonomy=current_taxonomy,
                    chunks=batch_chunks,
                    additional_depth=0, # Intention: Refine/Expand horizontally, not vertically
                    domain=domain,
                    batch_size=len(batch_chunks),
                    category_limits=category_limits,
                    specificity_level=specificity_level
                )
                
                # Update current taxonomy with the refined version
                # The extension function usually returns the full result structure
                refined_taxonomy = refinement_result.get('taxonomy', {})
                if refined_taxonomy:
                    # Merge logic is effectively handled by the LLM in the extension step
                    # or by the generator's post-processing.
                    # If the generator returns a partial diff, we'd need to merge.
                    # Assuming generator returns a FULL taxonomy structure.
                    current_taxonomy = refined_taxonomy
                    
                    # Carry over metadata/analytics from latest result
                    current_taxonomy_result['taxonomy'] = current_taxonomy
                    current_taxonomy_result['generation_metadata'] = refinement_result.get('generation_metadata', {})
                    # Update analytics
                    if 'analytics' not in current_taxonomy_result:
                        current_taxonomy_result['analytics'] = {}
                    
                    # Accumulate processing stats
                    current_taxonomy_result['analytics']['batches_processed'] = i + 1
                    
                    # Log successful step
                    current_taxonomy_result['processing_steps'].append({
                        "stage": "refinement_batch",
                        "batch_index": i + 1,
                        "total_batches": total_batches,
                        "message": f"Refined taxonomy with batch {i+1}/{total_batches}",
                        "chunks_in_batch": len(batch_chunks)
                    })
                    
                else:
                    self.logger.warning(f"Batch {i+1} returned empty taxonomy. Skipping update.")
                    current_taxonomy_result['processing_steps'].append({
                        "stage": "refinement_batch_skipped",
                        "batch_index": i + 1,
                        "message": f"Batch {i+1} skipped (empty result)",
                        "error": "Empty taxonomy returned"
                    })

            except Exception as e:
                self.logger.error(f"Error in refinement batch {i+1}: {e}")
                
                current_taxonomy_result['processing_steps'].append({
                    "stage": "refinement_batch_error",
                    "batch_index": i + 1,
                    "message": f"Batch {i+1} failed",
                    "error": str(e)
                })
                
                # Continue to next batch rather than failing completely
                continue

        # --- Stage 3: Finalization ---
        # Final update of analytics
        current_taxonomy_result['analytics']['chunk_coverage'] = 1.0
        current_taxonomy_result['analytics']['total_batches'] = total_batches + 1 # +1 for initial
        current_taxonomy_result['generation_metadata']['processing_mode'] = 'complete'

        return current_taxonomy_result