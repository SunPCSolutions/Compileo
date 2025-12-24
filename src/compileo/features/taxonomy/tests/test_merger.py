import unittest
from typing import Dict, Any, List
from src.compileo.features.taxonomy.merger import TaxonomyMerger
from src.compileo.features.extraction.context_models import HierarchicalCategory


class TestTaxonomyMerger(unittest.TestCase):

    def setUp(self):
        # Create a simple primary taxonomy
        self.primary_dict = {
            "name": "Root",
            "description": "Primary Root",
            "confidence_threshold": 0.8,
            "children": [
                {
                    "name": "Category A",
                    "description": "Primary A",
                    "confidence_threshold": 0.9,
                    "children": []
                },
                {
                    "name": "Category B",
                    "description": "Primary B",
                    "confidence_threshold": 0.7,
                    "children": [
                        {
                            "name": "Sub B1",
                            "description": "Primary B1",
                            "confidence_threshold": 0.8,
                            "children": []
                        }
                    ]
                }
            ]
        }

        # Create a secondary taxonomy to merge
        self.secondary_dict = {
            "name": "Root",
            "description": "Secondary Root",
            "confidence_threshold": 0.6,
            "children": [
                {
                    "name": "Category A", # Overlap
                    "description": "Secondary A (Better)",
                    "confidence_threshold": 0.95, # Higher confidence
                    "children": [
                         {
                            "name": "Sub A1", # New subcategory
                            "description": "Secondary A1",
                            "confidence_threshold": 0.85,
                            "children": []
                        }
                    ]
                },
                {
                    "name": "Category C", # New category
                    "description": "Secondary C",
                    "confidence_threshold": 0.75,
                    "children": []
                }
            ]
        }

    def test_merge_raw_taxonomies(self):
        merged = TaxonomyMerger.merge_raw_taxonomies(
            self.primary_dict, 
            self.secondary_dict,
            source1="primary", 
            source2="secondary"
        )
        
        # Check root
        self.assertEqual(merged["name"], "Merged Taxonomy")
        self.assertIn("primary", merged["description"])
        self.assertIn("secondary", merged["description"])
        
        # Check children count (A, B, C)
        self.assertEqual(len(merged["children"]), 3)
        child_names = [c["name"] for c in merged["children"]]
        self.assertIn("Category A", child_names)
        self.assertIn("Category B", child_names)
        self.assertIn("Category C", child_names)
        
        # Check enrichment of Category A
        cat_a = next(c for c in merged["children"] if c["name"] == "Category A")
        # Description should ideally be enriched if primary lacked one, but primary had one.
        # The logic is: "Improve description if primary lacks one". Primary had one, so it stays.
        self.assertEqual(cat_a["description"], "Primary A")
        # Confidence should be updated if secondary is higher
        self.assertEqual(cat_a["confidence_threshold"], 0.95)
        # Check merged_from
        self.assertIn("secondary", cat_a["merged_from"])
        
        # Check new subcategory in A
        self.assertEqual(len(cat_a["children"]), 1)
        self.assertEqual(cat_a["children"][0]["name"], "Sub A1")
        
        # Check preservation of Category B
        cat_b = next(c for c in merged["children"] if c["name"] == "Category B")
        self.assertEqual(cat_b["description"], "Primary B")
        self.assertEqual(len(cat_b["children"]), 1)
        
        # Check addition of Category C
        cat_c = next(c for c in merged["children"] if c["name"] == "Category C")
        self.assertEqual(cat_c["description"], "Secondary C")

    def test_flatten_and_rebuild(self):
        root = TaxonomyMerger._dict_to_category(self.primary_dict)
        flat = TaxonomyMerger._flatten_taxonomy(root, "test")
        
        # Expect paths: "Category A", "Category B", "Category B -> Sub B1"
        self.assertIn("Category A", flat)
        self.assertIn("Category B", flat)
        self.assertIn("Category B → Sub B1", flat)
        
        # Rebuild
        rebuilt_root = HierarchicalCategory(name="Rebuilt")
        TaxonomyMerger._rebuild_hierarchy(rebuilt_root, flat)
        
        rebuilt_dict = TaxonomyMerger._category_to_dict(rebuilt_root)
        self.assertEqual(len(rebuilt_dict["children"]), 2) # A and B

if __name__ == '__main__':
    unittest.main()