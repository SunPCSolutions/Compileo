import unittest
from unittest.mock import MagicMock, call
from src.compileo.features.taxonomy.iterative_generator import IterativeTaxonomyGenerator


class TestIterativeTaxonomyGenerator(unittest.TestCase):

    def setUp(self):
        # Mock generation function
        self.mock_generate = MagicMock()
        # Mock extension function
        self.mock_extend = MagicMock()
        
        self.generator = IterativeTaxonomyGenerator(
            generation_func=self.mock_generate,
            extension_func=self.mock_extend
        )
        
        # Test chunks
        self.chunks = [f"chunk{i}" for i in range(20)]

    def test_generate_single_batch(self):
        # Batch size covers all chunks (20 chunks, batch_size 25)
        
        # Mock return value for initial generation
        initial_result = {
            "taxonomy": {"name": "Test"},
            "generation_metadata": {},
            "analytics": {}
        }
        self.mock_generate.return_value = initial_result
        
        result = self.generator.generate(
            chunks=self.chunks,
            batch_size=25,
            domain="test"
        )
        
        # Should call generate once
        self.mock_generate.assert_called_once()
        # Should NOT call extend
        self.mock_extend.assert_not_called()
        
        # Check result
        self.assertEqual(result["taxonomy"]["name"], "Test")
        self.assertEqual(result["analytics"]["chunk_coverage"], 1.0)

    def test_generate_multi_stage(self):
        # 20 chunks, batch_size 5
        # Stage 1: Samples 5 chunks, generates baseline
        # Stage 2: 15 remaining chunks -> 3 batches of 5
        
        # Mock initial result
        initial_result = {
            "taxonomy": {"name": "Baseline"},
            "generation_metadata": {},
            "analytics": {}
        }
        self.mock_generate.return_value = initial_result
        
        # Mock extension results
        # We need to simulate that the taxonomy structure might change/grow
        # but for this test, we just return updated names to verify flow
        
        def extend_side_effect(**kwargs):
            # Return a slightly modified taxonomy each time
            existing = kwargs['existing_taxonomy']
            return {
                "taxonomy": {"name": existing["name"] + "+"},
                "generation_metadata": {},
                "analytics": {}
            }
            
        self.mock_extend.side_effect = extend_side_effect
        
        result = self.generator.generate(
            chunks=self.chunks,
            batch_size=5,
            domain="test"
        )
        
        # Verify calls
        self.mock_generate.assert_called_once()
        self.assertEqual(self.mock_extend.call_count, 3) # 15 remaining / 5 = 3 batches
        
        # Check that SmartChunkSampler was used (implicitly)
        # The generate call should have received 5 chunks
        args, _ = self.mock_generate.call_args
        # args[0] is 'chunks' if passed as positional, but we use keyword args in call
        # Let's check kwargs
        call_kwargs = self.mock_generate.call_args.kwargs
        self.assertEqual(len(call_kwargs['chunks']), 5)
        
        # Verify refinement flow
        # Final name should be "Baseline+++"
        self.assertEqual(result["taxonomy"]["name"], "Baseline+++")
        self.assertEqual(result["analytics"]["chunk_coverage"], 1.0)
        self.assertEqual(result["analytics"]["total_batches"], 4) # 1 initial + 3 refinement
        self.assertEqual(result["generation_metadata"]["processing_mode"], "complete")

    def test_error_handling_in_refinement(self):
        # Setup similar to multi-stage
        initial_result = {
            "taxonomy": {"name": "Baseline"},
            "generation_metadata": {},
            "analytics": {}
        }
        self.mock_generate.return_value = initial_result
        
        # Make extend fail on the first call, succeed on second
        self.mock_extend.side_effect = [
            Exception("API Error"),
            {
                "taxonomy": {"name": "Baseline+"},
                "generation_metadata": {},
                "analytics": {}
            }
        ]
        
        # 20 chunks, batch 5 -> 3 refinement batches
        # Batch 1: Fails
        # Batch 2: Succeeds
        # Batch 3: (Uses default mock return which is None if not set? No, side_effect iterator exhausted)
        # We need to provide enough side effects
        self.mock_extend.side_effect = [
            Exception("API Error"), # Batch 1 fails
            { # Batch 2 succeeds
                "taxonomy": {"name": "Baseline+"},
                "generation_metadata": {},
                "analytics": {}
            },
            { # Batch 3 succeeds
                "taxonomy": {"name": "Baseline++"},
                "generation_metadata": {},
                "analytics": {}
            }
        ]
        
        result = self.generator.generate(
            chunks=self.chunks,
            batch_size=5,
            domain="test"
        )
        
        # Should still complete and return result
        self.assertEqual(result["taxonomy"]["name"], "Baseline++")
        self.assertEqual(self.mock_extend.call_count, 3)

if __name__ == '__main__':
    unittest.main()