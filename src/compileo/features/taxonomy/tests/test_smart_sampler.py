import unittest
from src.compileo.features.taxonomy.smart_sampler import SmartChunkSampler


class TestSmartChunkSampler(unittest.TestCase):

    def test_sample_fewer_than_sample_size(self):
        chunks = ["chunk1", "chunk2", "chunk3"]
        sampled = SmartChunkSampler.sample(chunks, sample_size=5)
        self.assertEqual(len(sampled), 3)
        self.assertEqual(sampled, chunks)

    def test_sample_exact_sample_size(self):
        chunks = [f"chunk{i}" for i in range(5)]
        sampled = SmartChunkSampler.sample(chunks, sample_size=5)
        self.assertEqual(len(sampled), 5)
        self.assertEqual(sampled, chunks)

    def test_sample_more_than_sample_size(self):
        # 20 chunks, select 5
        chunks = [f"chunk{i}" for i in range(20)]
        sampled = SmartChunkSampler.sample(chunks, sample_size=5)
        
        self.assertEqual(len(sampled), 5)
        
        # Check if first and last are included
        self.assertEqual(sampled[0], "chunk0")
        self.assertEqual(sampled[-1], "chunk19")
        
        # Check if indices are somewhat distributed
        # Indices should be something like 0, 5, 10, 14, 19 depending on rounding
        indices = [int(c.replace("chunk", "")) for c in sampled]
        self.assertTrue(indices[0] == 0)
        self.assertTrue(indices[-1] == 19)
        self.assertTrue(all(indices[i] < indices[i+1] for i in range(len(indices)-1)))

    def test_sample_large_dataset(self):
        # 100 chunks, select 10
        chunks = [f"chunk{i}" for i in range(100)]
        sampled = SmartChunkSampler.sample(chunks, sample_size=10)
        
        self.assertEqual(len(sampled), 10)
        self.assertEqual(sampled[0], "chunk0")
        self.assertEqual(sampled[-1], "chunk99")

    def test_empty_input(self):
        chunks = []
        sampled = SmartChunkSampler.sample(chunks, sample_size=5)
        self.assertEqual(sampled, [])

if __name__ == '__main__':
    unittest.main()