import argparse
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate seed genes (ENSG) from TCGA mutation data."
    )

    parser.add_argument(
        "--cancer",
        type=str,
        required=True,
        help="Cancer type (e.g. BRCA, LUAD, CHOL)"
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=1.0,
        help="Minimum mutation frequency (%%). Default: 1.0"
    )

    return parser.parse_args()


def main():
    args = parse_args()

    input_file = f"./data_raw/Seed_raw/{args.cancer}.txt"
    mapping_file = "./data_set/ppi_network/mart_export.txt"
    output_file = f"./data_set/seed_set/TCGA-{args.cancer}_seed.txt"

    print(f"Loading: {input_file}")

    # Load mutation data
    df = pd.read_csv(input_file, sep="\t")

    # Filter cancer genes with frequency > threshold
    filtered = df[
        (
            df["Is Cancer Gene (source: OncoKB)"]
            .astype(str)
            .str.strip()
            .str.lower()
            == "yes"
        )
        &
        (
            df["Freq"]
            .str.rstrip("%")
            .astype(float)
            > args.threshold
        )
    ]

    seed_genes = filtered["Gene"].drop_duplicates().tolist()

    print(f"Selected genes: {len(seed_genes)}")

    # Load Gene -> ENSG mapping
    mapping = pd.read_csv(mapping_file, sep="\t")
    mapping.columns = mapping.columns.str.strip()

    gene_to_ensg = dict(
        zip(mapping["Gene name"], mapping["Gene stable ID"])
    )

    mapped_ensg = [
        gene_to_ensg[g]
        for g in seed_genes
        if g in gene_to_ensg
    ]

    # Save result
    with open(output_file, "w") as f:
        f.write("\n".join(mapped_ensg))

    print(f"Mapped ENSG genes: {len(mapped_ensg)}")
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()