#!/usr/bin/env python3
"""
Generate a complete dependency graph for the psy_supabase program.
This script analyzes all classes and their relationships, providing
a comprehensive view of the system architecture.
"""

import os
import re
import sys
import pydot
import inspect
import importlib
import pkgutil
from typing import List, Dict, Set, Tuple, Any

# Project root package
import psy_supabase

# Define patterns for finding class attributes and method calls
CLASS_PATTERN = r'class\s+(\w+)\s*(?:\(\s*(\w+)\s*\))?:'
METHOD_PATTERN = r'def\s+(\w+)\s*\('
ATTRIBUTE_PATTERN = r'self\.(\w+)\s*='
IMPORT_PATTERN = r'from\s+([\w\.]+)\s+import\s+([\w\s,]+)'
CALL_PATTERN = r'(\w+)\.(\w+)\s*\('
SELF_CALL_PATTERN = r'self\.(\w+)\s*\('

# Color mapping for different module categories
COLOR_MAP = {
    'core': '#E5F5E0',       # Light green
    'memory': '#FEE8C8',     # Light orange
    'utilities': '#DEEBF7',  # Light blue
    'external': '#E5E5E5',   # Light gray
    'unknown': '#FFFFFF'     # White
}

def get_module_color(module_name: str) -> str:
    """Determine color based on module category."""
    if 'core' in module_name:
        return COLOR_MAP['core']
    elif 'memory' in module_name:
        return COLOR_MAP['memory']
    elif 'utilities' in module_name or 'utils' in module_name:
        return COLOR_MAP['utilities']
    elif not module_name.startswith('psy_supabase'):
        return COLOR_MAP['external']
    else:
        return COLOR_MAP['unknown']

def discover_modules(package_name: str) -> List[str]:
    """Recursively discover all modules in a package."""
    package = importlib.import_module(package_name)
    modules = []

    for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
        modules.append(name)
        if is_pkg:
            modules.extend(discover_modules(name))

    return modules

def extract_classes_from_module(module_name: str) -> Dict[str, Dict[str, Any]]:
    """Extract class information from a module."""
    try:
        module = importlib.import_module(module_name)
        classes = {}

        for name, obj in inspect.getmembers(module, inspect.isclass):
            # Skip imported classes from other modules
            if obj.__module__ != module_name:
                continue

            try:
                source = inspect.getsource(obj)
                methods = re.findall(METHOD_PATTERN, source)
                attributes = re.findall(ATTRIBUTE_PATTERN, source)

                # Find parent class if any
                parent_match = re.search(CLASS_PATTERN, source)
                parent = parent_match.group(2) if parent_match and len(parent_match.groups()) > 1 else None

                # Get method dependencies
                method_deps = {}
                for method_name in methods:
                    if method_name.startswith('__'):
                        continue

                    try:
                        method_obj = getattr(obj, method_name)
                        method_source = inspect.getsource(method_obj)

                        # Find all self calls
                        self_calls = re.findall(SELF_CALL_PATTERN, method_source)

                        # Find all external calls
                        method_lines = method_source.split('\n')
                        external_calls = []

                        for line in method_lines:
                            # Skip comments
                            if '#' in line:
                                line = line[:line.index('#')]

                            matches = re.findall(CALL_PATTERN, line)
                            for match in matches:
                                if match[0] != 'self' and match[0] != 'cls':
                                    external_calls.append(f"{match[0]}.{match[1]}")

                        method_deps[method_name] = {
                            'self_calls': self_calls,
                            'external_calls': external_calls
                        }
                    except (TypeError, AttributeError):
                        continue

                classes[name] = {
                    'module': module_name,
                    'parent': parent,
                    'methods': methods,
                    'attributes': attributes,
                    'method_deps': method_deps
                }
            except (TypeError, OSError):
                # Skip classes without source code
                continue

        return classes
    except (ImportError, AttributeError) as e:
        print(f"Error importing module {module_name}: {e}")
        return {}

def build_program_graph() -> Dict[str, Dict[str, Any]]:
    """Build a complete program dependency graph."""
    # Discover all modules in the package
    modules = discover_modules('psy_supabase')

    # Extract classes from each module
    all_classes = {}
    module_to_classes = {}

    for module_name in modules:
        print(f"Analyzing module: {module_name}")
        classes = extract_classes_from_module(module_name)

        if classes:
            module_to_classes[module_name] = list(classes.keys())
            all_classes.update(classes)

    return {
        'classes': all_classes,
        'module_to_classes': module_to_classes
    }

def generate_class_graph(program_graph: Dict[str, Any]) -> pydot.Dot:
    """Generate a class dependency graph."""
    graph = pydot.Dot('psy_supabase_architecture', graph_type='digraph', rankdir='TB')

    # Set graph attributes for better visualization
    graph.set_graph_defaults(fontname='Arial', fontsize='16', splines='ortho')
    graph.set_node_defaults(
        shape='box',
        style='filled',
        fontname='Arial',
        fontsize='12',
        height='0.6',
        width='1.2'
    )

    # Create class nodes grouped by module
    class_nodes = {}
    module_clusters = {}

    # First, create module clusters
    for module, classes in program_graph['module_to_classes'].items():
        if not classes:
            continue

        # Extract module path for cluster name
        module_short = module.replace('psy_supabase.', '')
        cluster_name = f"cluster_{module_short.replace('.', '_')}"

        # Create cluster (subgraph) for the module
        cluster = pydot.Cluster(
            cluster_name,
            label=module_short,
            style='filled',
            fillcolor=get_module_color(module),
            color='gray70',
            fontname='Arial Bold',
            fontsize='14'
        )

        module_clusters[module] = cluster
        graph.add_subgraph(cluster)

    # Then, add class nodes to their respective clusters
    for class_name, class_info in program_graph['classes'].items():
        module_name = class_info['module']

        # Create node for this class
        node_label = f"{class_name}"
        if class_info['parent']:
            node_label += f"\\nExtends: {class_info['parent']}"

        node = pydot.Node(
            class_name,
            label=node_label,
            fillcolor='white',
            style='filled'
        )

        # Add to the correct module cluster
        if module_name in module_clusters:
            module_clusters[module_name].add_node(node)
        else:
            graph.add_node(node)

        class_nodes[class_name] = node

    # Add inheritance edges
    for class_name, class_info in program_graph['classes'].items():
        parent = class_info['parent']

        if parent and parent in class_nodes:
            edge = pydot.Edge(
                class_nodes[parent],
                class_nodes[class_name],
                arrowhead='empty',
                style='solid',
                weight='10',
                color='blue'
            )
            graph.add_edge(edge)

    # Add method call edges (composition/dependency)
    for class_name, class_info in program_graph['classes'].items():
        for method, deps in class_info['method_deps'].items():
            # Add edges for calls to other classes' methods
            for ext_call in deps['external_calls']:
                if '.' in ext_call:
                    parts = ext_call.split('.')
                    if len(parts) >= 2:
                        called_class, called_method = parts[0], parts[1]

                        # Only add edge if the called class exists in our graph
                        if called_class in class_nodes:
                            edge = pydot.Edge(
                                class_nodes[class_name],
                                class_nodes[called_class],
                                style='dashed',
                                color='gray50',
                                fontsize='9',
                                label=f" {method}()->{called_method}()"
                            )
                            graph.add_edge(edge)

    # Create legend
    legend = pydot.Cluster(
        'legend',
        label='Legend',
        fontsize='14',
        color='gray',
        style='filled',
        fillcolor='white'
    )

    legend_items = [
        ('Core Module', COLOR_MAP['core']),
        ('Memory Module', COLOR_MAP['memory']),
        ('Utilities Module', COLOR_MAP['utilities']),
        ('External Module', COLOR_MAP['external']),
        ('Inheritance', 'blue', 'solid'),
        ('Dependency', 'gray50', 'dashed')
    ]

    for i, item in enumerate(legend_items):
        if len(item) == 2:
            # Module type
            label, color = item
            node = pydot.Node(
                f'legend_{i}',
                label=label,
                shape='box',
                style='filled',
                fillcolor=color,
                fontsize='10'
            )
        else:
            # Relationship type
            label, color, style = item
            node = pydot.Node(
                f'legend_{i}',
                label=label,
                shape='plaintext',
                fontsize='10'
            )

        legend.add_node(node)

    graph.add_subgraph(legend)

    return graph

def generate_method_graph(class_name: str, program_graph: Dict[str, Any]) -> pydot.Dot:
    """Generate a method dependency graph for a specific class."""
    if class_name not in program_graph['classes']:
        print(f"Class {class_name} not found in program graph")
        return None

    class_info = program_graph['classes'][class_name]

    graph = pydot.Dot(f'{class_name}_methods', graph_type='digraph', rankdir='LR')

    # Set graph attributes
    graph.set_graph_defaults(fontname='Arial', fontsize='16')
    graph.set_node_defaults(
        shape='box',
        style='filled',
        fontname='Arial',
        fontsize='12',
        height='0.5',
        width='1.0'
    )

    # Create nodes for each method
    method_nodes = {}
    for method in class_info['methods']:
        if method.startswith('__'):
            continue

        # Style based on method type
        if method.startswith('_'):
            node = pydot.Node(
                f"{class_name}.{method}",
                label=method,
                fillcolor='#FEE6CE'  # Private methods
            )
        else:
            node = pydot.Node(
                f"{class_name}.{method}",
                label=method,
                fillcolor='#E5F5E0'  # Public methods
            )

        graph.add_node(node)
        method_nodes[method] = node

    # Add edges for method calls
    for method, deps in class_info['method_deps'].items():
        if method not in method_nodes:
            continue

        # Add edges for self calls
        for called_method in deps['self_calls']:
            if called_method in method_nodes:
                edge = pydot.Edge(
                    method_nodes[method],
                    method_nodes[called_method],
                    color='black',
                    fontsize='10'
                )
                graph.add_edge(edge)

    # Create legend
    legend = pydot.Cluster(
        'legend',
        label='Legend',
        fontsize='14',
        color='gray',
        style='filled',
        fillcolor='white'
    )

    legend_items = [
        ('Public Method', '#E5F5E0'),
        ('Private Method', '#FEE6CE')
    ]

    for i, (label, color) in enumerate(legend_items):
        node = pydot.Node(
            f'legend_{i}',
            label=label,
            shape='box',
            style='filled',
            fillcolor=color,
            fontsize='10'
        )
        legend.add_node(node)

    graph.add_subgraph(legend)

    return graph

def main():
    """Main function to generate the program graphs."""
    # Create output directory
    output_dir = "docs/diagrams"
    os.makedirs(output_dir, exist_ok=True)

    print("Building program dependency graph...")
    program_graph = build_program_graph()

    print(f"Found {len(program_graph['classes'])} classes in {len(program_graph['module_to_classes'])} modules")

    # Generate class-level graph
    print("Generating class-level dependency graph...")
    class_graph = generate_class_graph(program_graph)

    class_graph_path = f"{output_dir}/psy_supabase_classes.svg"
    class_graph.write_svg(class_graph_path)
    print(f"Class graph saved to {class_graph_path}")

    # Also save as PNG for easier viewing
    class_graph.write_png(f"{output_dir}/psy_supabase_classes.png")

    # Generate method-level graphs for key classes
    key_classes = [
        'DynamicRAGRetriever',
        'DatabaseManager',
        'ModelManager',
        'AssociativeMemory'
    ]

    for class_name in key_classes:
        if class_name in program_graph['classes']:
            print(f"Generating method graph for {class_name}...")
            method_graph = generate_method_graph(class_name, program_graph)

            if method_graph:
                method_graph_path = f"{output_dir}/{class_name}_methods.svg"
                method_graph.write_svg(method_graph_path)
                print(f"Method graph saved to {method_graph_path}")

                # Also save as PNG
                method_graph.write_png(f"{output_dir}/{class_name}_methods.png")
        else:
            print(f"Class {class_name} not found in program graph")

    print("\nAll graphs generated successfully!")
    print("You can find the diagrams in the docs/diagrams/ directory.")

if __name__ == "__main__":
    main()