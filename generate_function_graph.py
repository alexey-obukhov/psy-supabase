#!/usr/bin/env python3
"""
Generate a function dependency graph for the DynamicRAGRetriever class
and output as SVG.
"""

import os
import re
import pydot
import inspect
import importlib
from typing import List, Dict, Set, Tuple

# Import the class we want to analyze
from psy_supabase.core.dynamic_rag import DynamicRAGRetriever

def extract_method_calls(source_code: str, method_name: str) -> List[str]:
    """Extract method calls from a method's source code."""
    # Get class methods (excluding dunder methods)
    class_methods = [m for m in dir(DynamicRAGRetriever)
                     if callable(getattr(DynamicRAGRetriever, m))
                     and not m.startswith('__')]

    # Add self.xxx pattern to match method calls
    method_patterns = [rf'self\.{method}' for method in class_methods]

    # Find all instances of method calls in the source code
    calls = []
    for pattern in method_patterns:
        matches = re.findall(pattern + r'\s*\(', source_code)
        if matches:
            method = pattern.replace('self.', '')
            calls.append(method)

    # Also find external calls (to other components)
    external_patterns = [
        r'self\.db_manager\.(\w+)\s*\(',
        r'self\.associative_memory\.(\w+)\s*\(',
        r'self\.rag_processor\.(\w+)\s*\('
    ]

    external_calls = []
    for pattern in external_patterns:
        matches = re.findall(pattern, source_code)
        for match in matches:
            component = pattern.split('.')[1]  # Get the component name
            component = component.replace(r'(\w+)', '')  # Remove the regex pattern
            external_calls.append(f"{component}.{match}")

    return list(set(calls)), list(set(external_calls))

def build_dependency_graph() -> Tuple[Dict[str, List[str]], Dict[str, List[str]]]:
    """Build a dependency graph of method calls."""
    internal_dependencies = {}
    external_dependencies = {}

    # Get all method names
    methods = [m for m in dir(DynamicRAGRetriever)
               if callable(getattr(DynamicRAGRetriever, m))
               and not m.startswith('__')]

    for method in methods:
        # Get the method's source code
        method_obj = getattr(DynamicRAGRetriever, method)
        try:
            source_code = inspect.getsource(method_obj)
            internal_calls, external_calls = extract_method_calls(source_code, method)

            internal_dependencies[method] = internal_calls
            external_dependencies[method] = external_calls
        except (TypeError, OSError):
            # Skip methods without source code (e.g., built-ins)
            continue

    return internal_dependencies, external_dependencies

def generate_graph(internal_deps: Dict[str, List[str]], external_deps: Dict[str, List[str]]) -> pydot.Dot:
    """Generate a pydot graph from the dependency data."""
    graph = pydot.Dot('dynamic_rag_dependencies', graph_type='digraph', rankdir='LR')

    # Define node styling
    graph.set_node_defaults(
        shape='box',
        style='filled',
        fillcolor='#E5F5E0',
        fontname='Arial',
        fontsize='12'
    )

    # Create method nodes
    method_nodes = {}
    for method in internal_deps.keys():
        # Style nodes based on method type
        if method.startswith('_'):
            # Private methods
            node = pydot.Node(method, fillcolor='#FEE6CE')
        elif method == 'get_combined_retrieval' or method == 'get_combined_retrieval_workflow':
            # High-level methods
            node = pydot.Node(method, fillcolor='#C7E9C0', penwidth='2.0')
        else:
            # Standard public methods
            node = pydot.Node(method, fillcolor='#E5F5E0')

        graph.add_node(node)
        method_nodes[method] = node

    # Create external component nodes
    external_component_nodes = {}
    all_external_calls = set()
    for calls in external_deps.values():
        all_external_calls.update(calls)

    for call in all_external_calls:
        component = call.split('.')[0]
        if component not in external_component_nodes:
            node = pydot.Node(component, shape='ellipse', fillcolor='#DEEBF7', style='filled')
            graph.add_node(node)
            external_component_nodes[component] = node

    # Add edges for internal dependencies
    for method, calls in internal_deps.items():
        for called_method in calls:
            if called_method in method_nodes:
                graph.add_edge(pydot.Edge(method_nodes[method], method_nodes[called_method]))

    # Add edges for external dependencies
    for method, calls in external_deps.items():
        for external_call in calls:
            component = external_call.split('.')[0]
            if component in external_component_nodes:
                edge = pydot.Edge(
                    method_nodes[method],
                    external_component_nodes[component],
                    style='dashed',
                    color='#6BAED6'
                )
                graph.add_edge(edge)

    # Create legend
    legend = pydot.Cluster('legend', label='Legend', fontsize='14', color='gray')

    legend_items = [
        ('Public Method', '#E5F5E0'),
        ('Private Method', '#FEE6CE'),
        ('High-level Method', '#C7E9C0'),
        ('External Component', '#DEEBF7')
    ]

    for i, (label, color) in enumerate(legend_items):
        if label == 'External Component':
            node = pydot.Node(f'legend_{i}', label=label, shape='ellipse',
                             fillcolor=color, style='filled', fontsize='10')
        else:
            node = pydot.Node(f'legend_{i}', label=label, shape='box',
                             fillcolor=color, style='filled', fontsize='10')
        legend.add_node(node)

    graph.add_subgraph(legend)

    return graph

def main():
    """Main function to generate the SVG graph."""
    print("Analyzing DynamicRAGRetriever class...")
    internal_deps, external_deps = build_dependency_graph()

    print(f"Found {len(internal_deps)} methods with dependencies")

    print("Generating graph...")
    graph = generate_graph(internal_deps, external_deps)

    # Create output directory
    output_dir = "docs/diagrams"
    os.makedirs(output_dir, exist_ok=True)

    # Save as SVG
    output_path = f"{output_dir}/dynamic_rag_dependencies.svg"
    graph.write_svg(output_path)

    print(f"SVG graph saved to {output_path}")

    # Also save as PNG for easier viewing
    png_path = f"{output_dir}/dynamic_rag_dependencies.png"
    graph.write_png(png_path)
    print(f"PNG graph saved to {png_path}")

if __name__ == "__main__":
    main()