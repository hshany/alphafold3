#!/usr/bin/env python3
from rdkit import Chem
from rdkit.Chem import AllChem

BOND_ORDER_MAP = {
    Chem.BondType.SINGLE: "SING",
    Chem.BondType.DOUBLE: "DOUB",
    Chem.BondType.TRIPLE: "TRIP",
    Chem.BondType.AROMATIC: "AROM",
}

def default_atom_name(atom, idx0):
    # Simple, stable naming scheme: Element + 1-based index (C1, N5, S4, H11, ...)
    # You can override names for specific attachment atoms later if you want.
    return f"{atom.GetSymbol()}{idx0+1}"

def mol_to_ccd_cif(mol: Chem.Mol, comp_id: str, comp_name: str = None) -> str:
    if mol.GetNumConformers() == 0:
        raise ValueError("Input MOL has no 3D coordinates (no conformer).")

    conf = mol.GetConformer()
    comp_name = comp_name or comp_id

    # Atom naming (critical for AF3 bondedAtomPairs)
    atom_names = [default_atom_name(a, i) for i, a in enumerate(mol.GetAtoms())]

    # Header / _chem_comp
    lines = []
    lines.append(f"data_{comp_id}")
    lines.append(f"_chem_comp.id {comp_id}")
    lines.append(f"_chem_comp.name '{comp_name}'")
    lines.append(f"_chem_comp.type non-polymer")
    lines.append(f"_chem_comp.pdbx_synonyms ?")
    lines.append(f"_chem_comp.formula ?")
    lines.append(f"_chem_comp.mon_nstd_parent_comp_id ?")
    lines.append(f"_chem_comp.formula_weight ?")
    lines.append("")

    # _chem_comp_atom loop
    # Include ideal coords so AF3 can fall back if conformer generation fails. :contentReference[oaicite:2]{index=2}
    lines.append("loop_")
    lines.append("_chem_comp_atom.comp_id")
    lines.append("_chem_comp_atom.atom_id")
    lines.append("_chem_comp_atom.type_symbol")
    lines.append("_chem_comp_atom.charge")
    lines.append("_chem_comp_atom.pdbx_leaving_atom_flag")
    lines.append("_chem_comp_atom.pdbx_model_Cartn_x_ideal")
    lines.append("_chem_comp_atom.pdbx_model_Cartn_y_ideal")
    lines.append("_chem_comp_atom.pdbx_model_Cartn_z_ideal")

    for i, atom in enumerate(mol.GetAtoms()):
        pos = conf.GetAtomPosition(i)
        charge = atom.GetFormalCharge()
        # leaving_atom_flag: usually "N" for normal atoms
        lines.append(
            f"{comp_id} {atom_names[i]} {atom.GetSymbol()} {charge} N "
            f"{pos.x:.4f} {pos.y:.4f} {pos.z:.4f}"
        )
    lines.append("")

    # _chem_comp_bond loop
    lines.append("loop_")
    lines.append("_chem_comp_bond.comp_id")
    lines.append("_chem_comp_bond.atom_id_1")
    lines.append("_chem_comp_bond.atom_id_2")
    lines.append("_chem_comp_bond.value_order")
    lines.append("_chem_comp_bond.pdbx_aromatic_flag")
    lines.append("_chem_comp_bond.pdbx_stereo_config")
    lines.append("_chem_comp_bond.pdbx_ordinal")

    ordinal = 1
    for b in mol.GetBonds():
        a1 = b.GetBeginAtomIdx()
        a2 = b.GetEndAtomIdx()
        order = BOND_ORDER_MAP.get(b.GetBondType(), "sing")
        aromatic_flag = "Y" if b.GetIsAromatic() else "N"
        stereo = "N"  # keep simple; many ligands are fine with N
        lines.append(
            f"{comp_id} {atom_names[a1]} {atom_names[a2]} {order} {aromatic_flag} {stereo} {ordinal}"
        )
        ordinal += 1

    lines.append("")
    return "\n".join(lines)

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("mol_path")
    p.add_argument("--comp-id", default="LIG-1", help="CCD component ID (avoid underscores)")
    p.add_argument("--name", default=None, help="Human readable name")
    p.add_argument("-o", "--out", default="userccd.cif")
    p.add_argument(
        "--embed",
        action="store_true",
        help="Generate an RDKit ETKDG conformer before writing the CIF",
    )
    args = p.parse_args()

    mol = Chem.MolFromMolFile(args.mol_path, removeHs=False, sanitize=True)
    if mol is None:
        raise ValueError("RDKit failed to read MOL. Try sanitize=False or check file format.")

    if args.embed:
        # Re-embed with explicit hydrogens to stabilize geometry.
        mol = Chem.AddHs(mol)
        mol.RemoveAllConformers()
        status = AllChem.EmbedMolecule(mol, AllChem.ETKDG())
        if status != 0:
            raise RuntimeError("RDKit EmbedMolecule failed; inspect the input or tweak parameters.")

    cif = mol_to_ccd_cif(mol, comp_id=args.comp_id, comp_name=args.name)
    with open(args.out, "w") as f:
        f.write(cif)

    print(f"Wrote {args.out}")

if __name__ == "__main__":
    main()
