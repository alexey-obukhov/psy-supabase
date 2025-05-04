#!/usr/bin/env python3
"""
Generate a class communication diagram for the psy_supabase project.
This script visualizes how classes interact with each other through their methods.
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

# Color settings
MODULE_COLORS = {
    'core': '#E5F5E0',       # Light green
    'memory': '#FEE8C8',     # Light orange
    'utilities': '#DEEBF7',  # Light blue
    'database': '#F2E6FF',   # Light purple
    'external': '#E5E5E5',   # Light gray
    'default': '#FFFFFF'     # White
}

# Patterns for code analysis
IMPORT_PATTERN = r'from\s+([\w\.]+)\s+import\s+([\w\s,]+)'
CLASS_INSTANCE_PATTERN = r'self\.(\w+)\s*=\s*(\w+)\('
METHOD_CALL_PATTERN = r'self\.(\w+)\.(\w+)\s*\('

def discover_modules(package_name: str) -> List[str]:
    """Recursively discover all modules in a package."""
    package = importlib.import_module(package_name)
    modules = []

    for _, name, is_pkg in pkgutil.iter_modules(package.__path__, package.__name__ + '.'):
        modules.append(name)
        if is_pkg:
            modules.extend(discover_modules(name))

    return modules

def get_module_color(module_name: str) -> str:
    """Determine color based on module category."""
    if 'core' in module_name:
        return MODULE_COLORS['core']
    elif 'memory' in module_name:
        return MODULE_COLORS['memory']
    elif 'database' in module_name:
        return MODULE_COLORS['database']
    elif 'util' in module_name:
        return MODULE_COLORS['utilities']
    else:
        return MODULE_COLORS['default']

def extract_classes_and_communications(modules: List[str]) -> Dict[str, Any]:
    """Extract classes and their communications from modules."""
    classes = {}
    communications = []

    # First pass: collect all classes
    for module_name in modules:
        try:
            module = importlib.import_module(module_name)

            for name, obj in inspect.getmembers(module, inspect.isclass):
                # Skip imported classes
                if obj.__module__ != module_name:
                    continue

                classes[name] = {
                    'module': module_name,
                    'class': obj,
                    'attributes': [],
                    'methods': []
                }

                # Extract methods
                for method_name, method_obj in inspect.getmembers(obj, inspect.isfunction):
                    if not method_name.startswith('__'):
                        classes[name]['methods'].append(method_name)

        except ImportError as e:
            print(f"Error importing {module_name}: {e}")

    # Second pass: analyze communications
    for class_name, class_info in classes.items():
        try:
            # Get class source
            source = inspect.getsource(class_info['class'])

            # Find instance attributes that are other classes
            instance_matches = re.findall(CLASS_INSTANCE_PATTERN, source)
            for attr_name, attr_class in instance_matches:
                if attr_class in classes:
                    class_info['attributes'].append({
                        'name': attr_name,
                        'type': attr_class
                    })

                    # Add communication link (composition)
                    communications.append({
                        'from_class': class_name,
                        'to_class': attr_class,
                        'type': 'composition',
                        'methods': []
                    })

            # Find method calls to other class instances
            method_calls = re.findall(METHOD_CALL_PATTERN, source)
            for instance_name, method_name in method_calls:
                # Check which methods are called on which instances
                for attr in class_info['attributes']:
                    if attr['name'] == instance_name:
                        # Find existing communication or create new
                        comm_found = False
                        for comm in communications:
                            if (comm['from_class'] == class_name and
                                comm['to_class'] == attr['type'] and
                                comm['type'] == 'method_call'):

                                if method_name not in comm['methods']:
                                    comm['methods'].append(method_name)
                                comm_found = True
                                break

                        if not comm_found:
                            communications.append({
                                'from_class': class_name,
                                'to_class': attr['type'],
                                'type': 'method_call',
                                'methods': [method_name]
                            })

        except Exception as e:
            print(f"Error analyzing {class_name}: {e}")

    return {
        'classes': classes,
        'communications': communications
    }

def generate_communication_diagram(data: Dict[str, Any]) -> pydot.Dot:
    """Generate a class communication diagram."""
    graph = pydot.Dot('psy_supabase_communication', graph_type='digraph', rankdir='LR')

    # Configure graph appearance
    graph.set_graph_defaults(fontname='Arial', fontsize='16')
    graph.set_node_defaults(
        shape='box',
        style='filled',
        fontname='Arial',
        fontsize='12',
        height='0.6',
        margin='0.3,0.1'
    )

    # Group classes by module
    module_classes = {}
    for class_name, info in data['classes'].items():
        module = info['module']
        if module not in module_classes:
            module_classes[module] = []
        module_classes[module].append(class_name)

    # Create clusters for modules
    module_clusters = {}
    class_nodes = {}

    for module, classes in module_classes.items():
        module_short = module.replace('psy_supabase.', '')
        cluster_name = f"cluster_{module_short.replace('.', '_')}"

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

        # Add class nodes to cluster
        for class_name in classes:
            # Create a record-shaped node with methods
            methods = data['classes'][class_name]['methods']
            method_list = "|".join(methods[:5])  # Show top 5 methods

            if len(methods) > 5:
                method_list += "|..."

            node_label = f"{{<name> {class_name}|{method_list}}}"

            node = pydot.Node(
                class_name,
                label=node_label,
                shape='record',
                fillcolor='white'
            )

            cluster.add_node(node)
            class_nodes[class_name] = node

        graph.add_subgraph(cluster)

    # Add edges for communications
    for comm in data['communications']:
        from_class = comm['from_class']
        to_class = comm['to_class']

        if from_class not in class_nodes or to_class not in class_nodes:
            continue

        if comm['type'] == 'composition':
            # Composition relationship
            edge = pydot.Edge(
                class_nodes[from_class],
                class_nodes[to_class],
                arrowhead='diamond',
                style='solid',
                color='blue',
                weight='2'
            )
            graph.add_edge(edge)

        elif comm['type'] == 'method_call' and comm['methods']:
            # Method calls relationship
            methods_text = ", ".join(comm['methods'][:3])
            if len(comm['methods']) > 3:
                methods_text += ", ..."

            edge = pydot.Edge(
                class_nodes[from_class],
                class_nodes[to_class],
                label=f" {methods_text}",
                style='dashed',
                color='darkgreen',
                fontsize='9'
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

    # Create legend items
    legend_items = [
        ('Core Module', MODULE_COLORS['core']),
        ('Memory Module', MODULE_COLORS['memory']),
        ('Database Module', MODULE_COLORS['database']),
        ('Utilities Module', MODULE_COLORS['utilities']),
        ('Composition', 'blue', 'solid', 'diamond'),
        ('Method Calls', 'darkgreen', 'dashed', 'normal')
    ]

    legend_nodes = []
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
            legend.add_node(node)
            legend_nodes.append(node)
        else:
            # Relationship type
            label, color, style, arrowhead = item
            node = pydot.Node(
                f'legend_{i}',
                label=label,
                shape='plaintext',
                fontsize='10'
            )
            legend.add_node(node)
            legend_nodes.append(node)

    graph.add_subgraph(legend)

    # Create additional method detail diagram for key classes
    for class_name, class_info in data['classes'].items():
        if class_name in ['DynamicRAGRetriever', 'DatabaseManager', 'ModelManager', 'AssociativeMemory']:
            methods_used = set()

            # Find all methods used in communications
            for comm in data['communications']:
                if comm['to_class'] == class_name and comm['type'] == 'method_call':
                    methods_used.update(comm['methods'])

            # Create detailed node for this class with method call highlights
            if methods_used:
                method_details = []
                for method in class_info['methods']:
                    if method in methods_used:
                        method_details.append(f"<b>{method}</b>")
                    else:
                        method_details.append(method)

                # Only include this in a separate diagram

    return graph

def generate_focused_diagram(data: Dict[str, Any], focal_class: str) -> pydot.Dot:
    """Generate a diagram focused on a specific class and its communications."""
    graph = pydot.Dot(f'{focal_class}_communication', graph_type='digraph')

    # Configure graph appearance
    graph.set_graph_defaults(fontname='Arial', fontsize='16')
    graph.set_node_defaults(
        shape='box',
        style='filled',
        fontname='Arial',
        fontsize='12',
        height='0.6',
        margin='0.3,0.1'
    )

    # Find all classes that communicate with the focal class
    connected_classes = set([focal_class])

    for comm in data['communications']:
        if comm['from_class'] == focal_class:
            connected_classes.add(comm['to_class'])
        elif comm['to_class'] == focal_class:
            connected_classes.add(comm['from_class'])

    # Create nodes for all connected classes
    class_nodes = {}
    for class_name in connected_classes:
        if class_name not in data['classes']:
            continue

        class_info = data['classes'][class_name]
        module = class_info['module']

        # Create a record-shaped node with methods
        methods = class_info['methods']
        method_list = "|".join(methods[:7])  # Show more methods for focused view

        if len(methods) > 7:
            method_list += "|..."

        node_label = f"{{<name> {class_name}|{method_list}}}"

        fillcolor = get_module_color(module)
        if class_name == focal_class:
            # Highlight the focal class
            fillcolor = '#FFD700'  # Gold

        node = pydot.Node(
            class_name,
            label=node_label,
            shape='record',
            fillcolor=fillcolor,
            penwidth='2.0' if class_name == focal_class else '1.0'
        )

        graph.add_node(node)
        class_nodes[class_name] = node

    # Add edges for communications
    for comm in data['communications']:
        from_class = comm['from_class']
        to_class = comm['to_class']

        if (from_class in class_nodes and to_class in class_nodes and
            (from_class == focal_class or to_class == focal_class)):

            if comm['type'] == 'composition':
                # Composition relationship
                edge = pydot.Edge(
                    class_nodes[from_class],
                    class_nodes[to_class],
                    arrowhead='diamond',
                    style='solid',
                    color='blue',
                    weight='2'
                )
                graph.add_edge(edge)

            elif comm['type'] == 'method_call' and comm['methods']:
                # For focused view, show more method details
                methods_text = "\\n".join(comm['methods'][:10])
                if len(comm['methods']) > 10:
                    methods_text += "\\n..."

                edge = pydot.Edge(
                    class_nodes[from_class],
                    class_nodes[to_class],
                    label=methods_text,
                    style='dashed',
                    color='darkgreen',
                    fontsize='9'
                )
                graph.add_edge(edge)

    return graph

def main():
    """Main function to generate the communication diagrams."""
    # Create output directory
    output_dir = "docs/diagrams"
    os.makedirs(output_dir, exist_ok=True)

    print("Discovering modules...")
    modules = discover_modules('psy_supabase')
    print(f"Found {len(modules)} modules")

    print("Extracting classes and communications...")
    data = extract_classes_and_communications(modules)
    print(f"Found {len(data['classes'])} classes and {len(data['communications'])} communications")

    # Generate main communication diagram
    print("Generating main communication diagram...")
    main_diagram = generate_communication_diagram(data)

    main_diagram_path = f"{output_dir}/psy_supabase_communication.svg"
    main_diagram.write_svg(main_diagram_path)
    print(f"Main communication diagram saved to {main_diagram_path}")

    # Also save as PNG
    main_diagram.write_png(f"{output_dir}/psy_supabase_communication.png")

    # Generate focused diagrams for key classes
    key_classes = [
        'DynamicRAGRetriever',
        'DatabaseManager',
        'ModelManager',
        'AssociativeMemory',
        'TextGenerator'
    ]

    for class_name in key_classes:
        if class_name in data['classes']:
            print(f"Generating focused diagram for {class_name}...")
            focused_diagram = generate_focused_diagram(data, class_name)

            focused_diagram_path = f"{output_dir}/{class_name}_communication.svg"
            focused_diagram.write_svg(focused_diagram_path)
            print(f"Focused diagram saved to {focused_diagram_path}")

            # Also save as PNG
            focused_diagram.write_png(f"{output_dir}/{class_name}_communication.png")

    print("\nAll communication diagrams generated successfully!")
    print("You can find the diagrams in the docs/diagrams/ directory.")

if __name__ == "__main__":
    main()