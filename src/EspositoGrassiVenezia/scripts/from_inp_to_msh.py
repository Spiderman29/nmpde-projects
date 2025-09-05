def convert_inp_to_msh(nodes_file, elements_file, head_file, output_file):
    # Read nodes
    nodes = []
    with open(nodes_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 4:  # Valid node line
                node_id = int(parts[0])
                x = float(parts[1])
                y = float(parts[2])
                z = float(parts[3])
                nodes.append((node_id, x, y, z))
    
    # Read elements
    elements = []
    with open(elements_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 5:  # Valid element line
                elem_id = int(parts[0])
                n1 = int(parts[1])
                n2 = int(parts[2])
                n3 = int(parts[3])
                n4 = int(parts[4])
                elements.append((elem_id, n1, n2, n3, n4))
    
    # Get element sets
    gm_range = []
    wm_range = []
    with open(head_file, 'r') as f:
        content = f.read()
        
        # Look for the grey matter range
        gm_pattern = "*Elset, elset=gm, instance=PART-1-1, generate\n"
        if gm_pattern in content:
            gm_line_index = content.find(gm_pattern) + len(gm_pattern)
            gm_line = content[gm_line_index:].split('\n')[0].strip()
            gm_parts = gm_line.split(',')
            if len(gm_parts) >= 2:
                gm_range = [int(gm_parts[0]), int(gm_parts[1])]
        
        # Look for the white matter range
        wm_pattern = "*Elset, elset=wm, instance=PART-1-1, generate\n"
        if wm_pattern in content:
            wm_line_index = content.find(wm_pattern) + len(wm_pattern)
            wm_line = content[wm_line_index:].split('\n')[0].strip()
            wm_parts = wm_line.split(',')
            if len(wm_parts) >= 2:
                wm_range = [int(wm_parts[0]), int(wm_parts[1])]
        
        #Look for the cerebellum range
        cb_pattern = "*Elset, elset=cerebellum, instance=PART-1-1, generate\n"
        if cb_pattern in content:
            cb_line_index = content.find(cb_pattern) + len(cb_pattern)
            cb_line = content[cb_line_index:].split('\n')[0].strip()
            cb_parts = cb_line.split(',')
            if len(cb_parts) >= 2:
                cb_range = [int(cb_parts[0]), int(cb_parts[1])]
    
    print(f"GM Range: {gm_range}")
    print(f"WM Range: {wm_range}")
    print(f"Cerebellum Range: {cb_range}")

    # Filter elements for gm and wm
    filtered_elements = []
    for elem in elements:
        elem_id = elem[0]
        if (gm_range and gm_range[0] <= elem_id <= gm_range[1]) or (wm_range and wm_range[0] <= elem_id <= wm_range[1]) or (cb_range and cb_range[0] <= elem_id <= cb_range[1]):    
            filtered_elements.append(elem)
    
    print(f"Total elements: {len(elements)}")
    print(f"Filtered elements (GM+WM+CB): {len(filtered_elements)}")
    
    # Write MSH file (in Gmsh format)
    with open(output_file, 'w') as f:
        # MSH file header
        f.write("$MeshFormat\n")
        f.write("2.2 0 8\n")
        f.write("$EndMeshFormat\n")
        
        # Nodes section
        f.write("$Nodes\n")
        f.write(f"{len(nodes)}\n")
        for node in nodes:
            f.write(f"{node[0]} {node[1]} {node[2]} {node[3]}\n")
        f.write("$EndNodes\n")
        
        # Elements section
        f.write("$Elements\n")
        f.write(f"{len(filtered_elements)}\n")
        
        for elem in filtered_elements:
            # For tetrahedra in Gmsh format: elem_number elm_type tags... node_indices...
            # elm_type 4 is for 4-node tetrahedra
            # Using 2 tags: physical group and element set
            physical_group = 1  # Default
            if gm_range and gm_range[0] <= elem[0] <= gm_range[1]:
                element_set = 1  # GM
            elif wm_range and wm_range[0] <= elem[0] <= wm_range[1]:
                element_set = 2  # WM
            else:
                element_set = 3  # Cerebellum
            physical_group = element_set  # Use element set (1, 2, or 3) as physical group
            f.write(f"{elem[0]} 4 2 {physical_group} {element_set} {elem[1]} {elem[2]} {elem[3]} {elem[4]}\n")
        
        f.write("$EndElements\n")

# Example usage
convert_inp_to_msh("nodes.inp", "elements_abq.inp", "brain_mesh_CMAME.inp", "brain_gm_wm_cb.msh")