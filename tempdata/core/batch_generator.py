"""
Batch generator for managing related dataset generation

Provides BatchGenerator class to manage related dataset generation with
referential integrity maintenance across datasets.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
import pandas as pd
from dataclasses import dataclass
import uuid
from pathlib import Path

from .seeding import MillisecondSeeder
from ..exporters.export_manager import ExportManager


@dataclass
class DatasetSpec:
    """Specification for a dataset in a batch"""
    name: str
    filename: str
    rows: int
    dataset_type: str
    relationships: List[str] = None  # List of dataset names this depends on
    custom_params: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.relationships is None:
            self.relationships = []
        if self.custom_params is None:
            self.custom_params = {}


@dataclass
class RelationshipSpec:
    """Specification for a relationship between datasets"""
    source_dataset: str
    target_dataset: str
    source_column: str
    target_column: str
    relationship_type: str  # 'one_to_many', 'many_to_one', 'one_to_one'
    cascade_delete: bool = False


class BatchGenerator:
    """
    Manager class for related dataset generation with referential integrity
    
    Handles the generation of multiple related datasets while maintaining
    relationships and consistency across datasets.
    """
    
    def __init__(self, base_seeder: MillisecondSeeder):
        """
        Initialize batch generator
        
        Args:
            base_seeder: Base seeder for consistent relationships
        """
        self.base_seeder = base_seeder
        self.datasets: Dict[str, DatasetSpec] = {}
        self.relationships: List[RelationshipSpec] = []
        self.generated_data: Dict[str, pd.DataFrame] = {}
        self.export_manager = ExportManager()
        
        # Relationship tracking
        self._dependency_graph: Dict[str, Set[str]] = {}
        self._generation_order: List[str] = []
        
    def add_dataset(self, spec: DatasetSpec) -> None:
        """
        Add a dataset specification to the batch
        
        Args:
            spec: Dataset specification
            
        Raises:
            ValueError: If dataset name already exists
        """
        if spec.name in self.datasets:
            raise ValueError(f"Dataset '{spec.name}' already exists in batch")
        
        self.datasets[spec.name] = spec
        self._dependency_graph[spec.name] = set(spec.relationships)
        
    def add_relationship(self, relationship: RelationshipSpec) -> None:
        """
        Add a relationship specification between datasets
        
        Args:
            relationship: Relationship specification
            
        Raises:
            ValueError: If source or target dataset doesn't exist
        """
        if relationship.source_dataset not in self.datasets:
            raise ValueError(f"Source dataset '{relationship.source_dataset}' not found")
        
        if relationship.target_dataset not in self.datasets:
            raise ValueError(f"Target dataset '{relationship.target_dataset}' not found")
        
        self.relationships.append(relationship)
        
        # Update dependency graph
        self._dependency_graph[relationship.target_dataset].add(relationship.source_dataset)
    
    def _validate_batch_configuration(self) -> None:
        """
        Validate the batch configuration for consistency
        
        Raises:
            ValueError: If configuration is invalid
        """
        # Check for circular dependencies
        if self._has_circular_dependencies():
            raise ValueError("Circular dependencies detected in dataset relationships")
        
        # Validate relationship specifications
        for rel in self.relationships:
            source_spec = self.datasets[rel.source_dataset]
            target_spec = self.datasets[rel.target_dataset]
            
            # Validate relationship types
            valid_types = ['one_to_many', 'many_to_one', 'one_to_one']
            if rel.relationship_type not in valid_types:
                raise ValueError(f"Invalid relationship type: {rel.relationship_type}")
    
    def _has_circular_dependencies(self) -> bool:
        """
        Check if there are circular dependencies in the dataset relationships
        
        Returns:
            bool: True if circular dependencies exist
        """
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for neighbor in self._dependency_graph.get(node, set()):
                if neighbor not in visited:
                    if has_cycle(neighbor):
                        return True
                elif neighbor in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for dataset in self.datasets:
            if dataset not in visited:
                if has_cycle(dataset):
                    return True
        
        return False
    
    def _calculate_generation_order(self) -> List[str]:
        """
        Calculate the order in which datasets should be generated
        
        Returns:
            List[str]: Ordered list of dataset names
        """
        # Topological sort to determine generation order
        in_degree = {name: len(deps) for name, deps in self._dependency_graph.items()}
        queue = [name for name, degree in in_degree.items() if degree == 0]
        result = []
        
        while queue:
            current = queue.pop(0)
            result.append(current)
            
            # Update in-degrees of dependent datasets
            for dataset, deps in self._dependency_graph.items():
                if current in deps:
                    in_degree[dataset] -= 1
                    if in_degree[dataset] == 0:
                        queue.append(dataset)
        
        if len(result) != len(self.datasets):
            raise ValueError("Unable to determine generation order - circular dependencies detected")
        
        return result
    
    def _generate_single_dataset(self, dataset_name: str, **global_params) -> pd.DataFrame:
        """
        Generate a single dataset with relationship constraints
        
        Args:
            dataset_name: Name of dataset to generate
            **global_params: Global parameters for generation
            
        Returns:
            pd.DataFrame: Generated dataset
        """
        spec = self.datasets[dataset_name]
        
        # Import the appropriate generator
        from ..api import DATASET_GENERATORS
        from ..core.localization import LocalizationEngine
        
        if spec.dataset_type not in DATASET_GENERATORS:
            raise ValueError(f"Unsupported dataset type: {spec.dataset_type}")
        
        # Get proper locale
        country = global_params.get('country', 'global')
        if country == 'global':
            country = 'united_states'
        
        localization = LocalizationEngine()
        locale = localization.get_locale(country)
        
        # Create contextual seeder for this dataset
        contextual_seed = self.base_seeder.get_contextual_seed(f"batch_{dataset_name}")
        dataset_seeder = MillisecondSeeder(fixed_seed=contextual_seed)
        
        # Initialize generator
        generator_class = DATASET_GENERATORS[spec.dataset_type]
        generator = generator_class(dataset_seeder, locale)
        
        # Merge global and dataset-specific parameters
        generation_params = global_params.copy()
        generation_params.update(spec.custom_params)
        
        # Apply relationship constraints
        relationship_constraints = self._get_relationship_constraints(dataset_name)
        generation_params.update(relationship_constraints)
        
        # Remove 'rows' from generation_params to avoid duplicate argument
        generation_params.pop('rows', None)
        
        # Generate data
        data = generator.generate(spec.rows, **generation_params)
        
        # Apply relationship integrity
        data = self._apply_relationship_integrity(dataset_name, data)
        
        return data
    
    def _get_relationship_constraints(self, dataset_name: str) -> Dict[str, Any]:
        """
        Get relationship constraints for a dataset
        
        Args:
            dataset_name: Name of dataset
            
        Returns:
            Dict[str, Any]: Relationship constraints
        """
        constraints = {}
        
        # Find relationships where this dataset is the target
        for rel in self.relationships:
            if rel.target_dataset == dataset_name:
                source_data = self.generated_data.get(rel.source_dataset)
                if source_data is not None and rel.source_column in source_data.columns:
                    # Provide reference values for foreign key generation
                    reference_values = source_data[rel.source_column].unique().tolist()
                    constraints[f"{rel.target_column}_references"] = reference_values
                    constraints[f"{rel.target_column}_relationship_type"] = rel.relationship_type
        
        return constraints
    
    def _apply_relationship_integrity(self, dataset_name: str, data: pd.DataFrame) -> pd.DataFrame:
        """
        Apply relationship integrity constraints to generated data
        
        Args:
            dataset_name: Name of dataset
            data: Generated data
            
        Returns:
            pd.DataFrame: Data with relationship integrity applied
        """
        # Find relationships where this dataset is the target
        for rel in self.relationships:
            if rel.target_dataset == dataset_name:
                source_data = self.generated_data.get(rel.source_dataset)
                
                if source_data is not None and rel.source_column in source_data.columns:
                    reference_values = source_data[rel.source_column].unique().tolist()
                    
                    if rel.target_column in data.columns:
                        # Apply relationship constraints based on type
                        import random
                        
                        # Set random seed for consistent relationship generation
                        random.seed(self.base_seeder.get_contextual_seed(f"relationship_{rel.target_dataset}"))
                        
                        if rel.relationship_type == 'one_to_many':
                            # Each target record should reference a source record
                            data[rel.target_column] = [
                                random.choice(reference_values) for _ in range(len(data))
                            ]
                        elif rel.relationship_type == 'many_to_one':
                            # Multiple target records can reference the same source
                            data[rel.target_column] = [
                                random.choice(reference_values) for _ in range(len(data))
                            ]
                        elif rel.relationship_type == 'one_to_one':
                            # Each target record should reference a unique source record
                            if len(data) <= len(reference_values):
                                selected_refs = random.sample(reference_values, len(data))
                                data[rel.target_column] = selected_refs
                            else:
                                # If more target records than source, some will be duplicated
                                data[rel.target_column] = [
                                    reference_values[i % len(reference_values)] 
                                    for i in range(len(data))
                                ]
        
        return data
    
    def _validate_relationships(self) -> List[str]:
        """
        Validate that all relationships are properly maintained
        
        Returns:
            List[str]: List of validation errors (empty if all valid)
        """
        errors = []
        
        for rel in self.relationships:
            source_data = self.generated_data.get(rel.source_dataset)
            target_data = self.generated_data.get(rel.target_dataset)
            
            if source_data is None or target_data is None:
                continue
            
            if rel.source_column not in source_data.columns:
                errors.append(f"Source column '{rel.source_column}' not found in {rel.source_dataset}")
                continue
            
            if rel.target_column not in target_data.columns:
                errors.append(f"Target column '{rel.target_column}' not found in {rel.target_dataset}")
                continue
            
            # Check referential integrity
            source_values = set(source_data[rel.source_column].unique())
            target_values = set(target_data[rel.target_column].unique())
            
            # All target values should exist in source
            orphaned_values = target_values - source_values
            if orphaned_values:
                errors.append(
                    f"Orphaned values in {rel.target_dataset}.{rel.target_column}: {orphaned_values}"
                )
            
            # Validate relationship cardinality
            if rel.relationship_type == 'one_to_one':
                target_counts = target_data[rel.target_column].value_counts()
                duplicates = target_counts[target_counts > 1]
                if not duplicates.empty:
                    errors.append(
                        f"One-to-one relationship violated in {rel.target_dataset}.{rel.target_column}"
                    )
        
        return errors
    
    def generate_batch(self, **global_params) -> Dict[str, str]:
        """
        Generate all datasets in the batch with maintained relationships
        
        Args:
            **global_params: Global parameters applied to all datasets
            
        Returns:
            Dict[str, str]: Mapping of dataset names to generated file paths
            
        Raises:
            ValueError: If batch configuration is invalid
            IOError: If generation fails
        """
        # Validate configuration
        self._validate_batch_configuration()
        
        # Calculate generation order
        self._generation_order = self._calculate_generation_order()
        
        results = {}
        
        try:
            # Generate datasets in dependency order
            for dataset_name in self._generation_order:
                spec = self.datasets[dataset_name]
                
                # Generate dataset
                data = self._generate_single_dataset(dataset_name, **global_params)
                self.generated_data[dataset_name] = data
                
                # Export dataset - check for dataset-specific formats first
                dataset_formats = spec.custom_params.get('formats')
                if dataset_formats is None:
                    formats = global_params.get('formats', ['csv'])
                else:
                    formats = dataset_formats
                
                base_filename = Path(spec.filename).stem
                
                if len(formats) == 1:
                    result_path = self.export_manager.export_single(data, base_filename, formats[0])
                else:
                    result_paths = self.export_manager.export_multiple(data, base_filename, formats)
                    result_path = ', '.join(result_paths.values())
                
                results[dataset_name] = result_path
            
            # Validate relationships after all generation
            validation_errors = self._validate_relationships()
            if validation_errors:
                error_msg = "Relationship validation failed:\n" + "\n".join(validation_errors)
                raise ValueError(error_msg)
            
            return results
            
        except Exception as e:
            # Clean up any generated files on failure
            self._cleanup_generated_files(results)
            raise IOError(f"Batch generation failed: {str(e)}")
    
    def _cleanup_generated_files(self, file_paths: Dict[str, str]) -> None:
        """
        Clean up generated files on failure
        
        Args:
            file_paths: Dictionary of generated file paths
        """
        import os
        
        for path_str in file_paths.values():
            # Handle comma-separated paths (multiple formats)
            paths = path_str.split(', ') if ', ' in path_str else [path_str]
            
            for path in paths:
                try:
                    if os.path.exists(path):
                        os.remove(path)
                except OSError:
                    pass  # Ignore cleanup errors
    
    def get_relationship_summary(self) -> Dict[str, Any]:
        """
        Get a summary of relationships in the batch
        
        Returns:
            Dict[str, Any]: Relationship summary
        """
        return {
            'total_datasets': len(self.datasets),
            'total_relationships': len(self.relationships),
            'generation_order': self._generation_order,
            'relationships': [
                {
                    'source': rel.source_dataset,
                    'target': rel.target_dataset,
                    'type': rel.relationship_type,
                    'columns': f"{rel.source_column} -> {rel.target_column}"
                }
                for rel in self.relationships
            ]
        }
    
    def validate_batch_integrity(self) -> Dict[str, Any]:
        """
        Validate the integrity of the generated batch
        
        Returns:
            Dict[str, Any]: Validation results
        """
        if not self.generated_data:
            return {'status': 'no_data', 'message': 'No data generated yet'}
        
        validation_errors = self._validate_relationships()
        
        return {
            'status': 'valid' if not validation_errors else 'invalid',
            'errors': validation_errors,
            'datasets_generated': len(self.generated_data),
            'relationships_validated': len(self.relationships)
        }